import re
import os
import json
import yaml
from datetime import datetime
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from google.analytics.data_v1beta.types import RunReportRequest
from typing import List, Optional, Dict, Literal, Type, Any
from pydantic import BaseModel, Field, field_validator, ValidationError, constr, ConfigDict, PrivateAttr


# Метрика



class GA4_Chat_Answer:
    def __init__(self, client_ga, ai_model = 'gpt-4o-mini', ga4_property = 281955181):
        self.ga4_property = ga4_property
        self.client_ga = client_ga
        self.metadata = self.client_ga.get_metadata(name=f"properties/{self.ga4_property}/metadata")
        self.metrics_list = [m.api_name for m in self.metadata.metrics]
        self.dimensions_list = [d.api_name for d in self.metadata.dimensions]
        self.metrics1 = ",\n".join(self.metrics_list)
        self.dimensions1 = ",\n".join(self.dimensions_list)

        class BaseModelF(BaseModel):
            """Базовый класс для всех моделей с запретом дополнительных параметров."""
            model_config = ConfigDict(extra="forbid")


        class Metric(BaseModelF):
            name: Literal[*self.metrics_list] = Field(..., description="Name of the metric (e.g., 'sessions', 'engagementRate')")

        # Измерение
        class Dimension(BaseModelF):
            name: Literal[*self.dimensions_list] = Field(...,description="Name of the dimension (e.g., 'country', 'deviceCategory')")

        # Диапазон дат
        class DateRange(BaseModelF):
            """Defines a date range for filtering data in GA4 queries."""

            start_date: str = Field(..., description="The start date of the range in 'yyyy-mm-dd' format. This date is inclusive, meaning data from this day will be included in the results.")
            end_date: str = Field(..., description="The end date of the range in 'yyyy-mm-dd' format. This date is also inclusive, so data from this day will be included in the results.")

        # Фильтр по строкам
        class StringFilter(BaseModel, extra='forbid'):
            match_type: Literal['MATCH_TYPE_UNSPECIFIED','EXACT','BEGINS_WITH','ENDS_WITH','CONTAINS','FULL_REGEXP','PARTIAL_REGEXP'] = Field(...,description="The type of string matching to apply:\n"
                    "- 'EXACT': Matches the string exactly.\n"
                    "- 'BEGINS_WITH': Matches strings that start with the given value.\n"
                    "- 'ENDS_WITH': Matches strings that end with the given value.\n"
                    "- 'CONTAINS': Matches strings that contain the given value.\n"
                    "- 'FULL_REGEXP': Matches using a full regular expression.\n"
                    "- 'PARTIAL_REGEXP': Matches using a partial regular expression.\n"
                    "Use 'MATCH_TYPE_UNSPECIFIED' if the match type is unknown.")
            value: str = Field(..., description="The string value to match against the specified field. Behavior depends on the selected match type.")
            case_sensitive: Optional[bool] = Field(False, description="If True, the matching is case-sensitive. If False (default), matching ignores letter case.")


        # Числовое значение для фильтра
        class NumericValue(BaseModelF):
            """Represents a numeric value that can be used in filters, supporting both integer and floating-point numbers."""

            int64_value: Optional[int] = Field(None, description="An integer value for the numeric filter. Use this when filtering whole numbers, such as user counts or session counts.")
            double_value: Optional[float] = Field(None, description="A floating-point value for the numeric filter. Use this when filtering decimal values, such as revenue or engagement time.")

        # Числовой фильтр
        class NumericFilter(BaseModelF):
            """Defines a numeric filter for comparing a metric's value against a specified threshold."""
            operation: Literal['OPERATION_UNSPECIFIED','EQUAL','LESS_THAN','LESS_THAN_OR_EQUAL','GREATER_THAN','GREATER_THAN_OR_EQUAL'] = Field(..., description="The comparison operation to apply:\n"
                    "- 'EQUAL': Matches values exactly equal to the given number.\n"
                    "- 'LESS_THAN': Matches values strictly less than the given number.\n"
                    "- 'LESS_THAN_OR_EQUAL': Matches values less than or equal to the given number.\n"
                    "- 'GREATER_THAN': Matches values strictly greater than the given number.\n"
                    "- 'GREATER_THAN_OR_EQUAL': Matches values greater than or equal to the given number.\n"
                    "Use 'OPERATION_UNSPECIFIED' if no specific operation is defined.")
            value: NumericValue = Field(..., description="The numeric value to compare against the metric. For example, filtering users with revenue 'GREATER_THAN' 100.")

        class InListFilter(BaseModelF):
            """Filters results based on whether a field's value is present in a predefined list of values."""

            values: List[str] = Field(...,description="A list of allowed values. The field will match if its value is present in this list. For example, to filter by specific countries, use ['USA', 'Canada', 'Germany'].")
            case_sensitive: Optional[bool] = Field(False,description="If True, matching will be case-sensitive (e.g., 'USA' and 'usa' would be treated as different values). If False (default), matching ignores letter case.")

        class BetweenFilter(BaseModelF):
            """Defines a numeric range filter to match values within a specified range (inclusive)."""

            from_value: NumericValue = Field(None, description="The lower bound of the range (inclusive). Values must be greater than or equal to this number.")
            to_value: NumericValue = Field(None, description="The upper bound of the range (inclusive). Values must be less than or equal to this number.")

        class Filter(BaseModelF):
            """Represents a filter for GA4 queries, allowing different types of conditions on dimensions and metrics."""

            field_name: Literal[*self.metrics_list,*self.dimensions_list] = Field(...,description="The name of the field (dimension or metric) to filter. For example, 'country' for dimensions or 'totalRevenue' for metrics.")

            string_filter: Optional[StringFilter] = Field(None, description="A string-based filter that allows matching field values using exact, partial, or regex patterns. Use this for text-based filtering.")

            numeric_filter: Optional[NumericFilter] = Field(None,description="A numeric filter to compare metric values using operators like greater than, less than, or equals.")

            in_list_filter: Optional[InListFilter] = Field(None,description="A filter that allows matching field values against a predefined list. For example, filtering by a specific set of countries.")

            between_filter: Optional[BetweenFilter] = Field(None,description="A filter to match numeric values within a specified range. For example, selecting revenue between $100 and $500.")

            empty_filter: Optional[List[str]] = Field(None,description="A list of fields that should be considered as empty. Useful for filtering out missing or null values.")

        # Параметры сортировки

        class MetricOrderBy(BaseModelF):
            """Specifies sorting by a metric in GA4 reports."""

            metric_name: Literal[*self.metrics_list] = Field(..., description="The name of the metric to order by (e.g., 'totalRevenue', 'sessions').")

        class DimensionOrderBy(BaseModelF):
            """Specifies sorting by a dimension in GA4 reports, with different ordering strategies."""

            dimension_name: Literal[*self.dimensions_list] = Field(..., description="The name of the dimension to order by (e.g., 'country', 'deviceCategory').")
            order_type: Optional[Literal["ORDER_TYPE_UNSPECIFIED", "ALPHANUMERIC", "CASE_INSENSITIVE_ALPHANUMERIC", "NUMERIC"]] = Field(None,
        description="Specifies how to sort dimension values:\n"
                    "- 'ALPHANUMERIC': Case-sensitive alphanumeric sorting.\n"
                    "- 'CASE_INSENSITIVE_ALPHANUMERIC': Alphanumeric sorting, ignoring case.\n"
                    "- 'NUMERIC': Sorts values as numbers.\n"
                    "- 'ORDER_TYPE_UNSPECIFIED': Default sorting behavior.")

        class PivotSelection(BaseModelF):
            """Defines how to filter pivot table data based on a specific dimension value."""

            dimension_name: Literal[*self.dimensions_list] = Field(...,description="The name of the dimension to filter within the pivot table (e.g., 'country').")
            dimension_value: str = Field(...,description="The specific value of the dimension to filter by (e.g., 'Germany').")

        class PivotOrderBy(BaseModelF):
            """Defines sorting rules for pivot tables, ordering metrics within specific pivot selections."""

            metric_name: Literal[*self.metrics_list] = Field(...,description="The name of the metric to order pivot table data by (e.g., 'totalRevenue').")
            pivot_selections: List[PivotSelection] = Field(...,description="A list of pivot selections that define how data is filtered before sorting.")

        class OrderByItem(BaseModelF):
            """Specifies sorting rules for the query results in GA4 reports."""

            metric: Optional[MetricOrderBy] = Field(None,description="Sorting criteria based on a metric value. Used when ordering results by numerical data (e.g., revenue, session count).")
            dimension: Optional[DimensionOrderBy] = Field(None,description="Sorting criteria based on a dimension value. Used when ordering results by categorical data (e.g., country, device type).")
            pivot: Optional[PivotOrderBy] = Field(None,description="Sorting criteria for pivot tables, allowing ordering of pivot dimensions or metrics.")
            desc: bool = Field(False,description="If True, the results will be sorted in descending order. If False (default), sorting is ascending.")

        # --------------------------
        class AndGroup_sub3(BaseModelF):
            expressions: Optional[List[str]] = None

        class OrGroup_sub3(BaseModelF):
            expressions: Optional[List[str]] = None



        # Выражение фильтра
        class FilterExpression_sub3(BaseModelF):
            filter: Optional[Filter] = None
            and_group: Optional[AndGroup_sub3] = None
            or_group: Optional[OrGroup_sub3] = None
            not_expression: Optional[Dict[str,Any]] = None

        # --------------------------3
        class AndGroup_sub2(BaseModelF):
            expressions: Optional[List[FilterExpression_sub3]] = None

        class OrGroup_sub2(BaseModelF):
            expressions: Optional[List[FilterExpression_sub3]] = None

        # Выражение фильтра
        class FilterExpression_sub2(BaseModelF):
            filter: Optional[Filter] = None
            and_group: Optional[AndGroup_sub2] = None
            or_group: Optional[OrGroup_sub2] = None
            not_expression: Optional[FilterExpression_sub3] = None

        # --------------------------2
        class AndGroup_sub1(BaseModelF):
            expressions: Optional[List[FilterExpression_sub2]] = None

        class OrGroup_sub1(BaseModelF):
            expressions: Optional[List[FilterExpression_sub2]] = None

        # Выражение фильтра
        class FilterExpression_sub1(BaseModelF):
            filter: Optional[Filter] = None
            and_group: Optional[AndGroup_sub1] = None
            or_group: Optional[OrGroup_sub1] = None
            not_expression: Optional[FilterExpression_sub2] = None

        # --------------------------1
        class AndGroup(BaseModelF):
            """Represents a group of filter expressions combined using a logical AND operation."""

            expressions: Optional[List[FilterExpression_sub1]] = Field(None,description="A list of filter expressions that must all be satisfied for a record to match. If multiple expressions are provided, all must evaluate to True for the filter to apply.")

        class OrGroup(BaseModelF):
            """Represents a group of filter expressions combined using a logical OR operation."""

            expressions: Optional[List[FilterExpression_sub1]] = Field(None,description="A list of filter expressions where at least one must be satisfied for a record to match. If multiple expressions are provided, any single one evaluating to True will apply the filter.")



        # Выражение фильтра
        class FilterExpression(BaseModelF):
            """Defines a filter expression that can combine multiple filtering conditions using logical operators."""

            filter: Optional[Filter] = Field(None, description='''A single filtering condition applied to a specific field. All filters in the same expression must target either dimensions or metrics, but not both.''')
            and_group: Optional[AndGroup] = Field(None, description='''A group of filters combined with AND logic. All conditions within this group must be met for the data to be included.''')
            or_group: Optional[OrGroup] = Field(None, description='''A group of filters combined with OR logic. At least one condition within this group must be met for the data to be included.''')
            not_expression: Optional[FilterExpression_sub1] = Field(None, description='''Negates the given filter expression. Data that matches this condition will be excluded from the results.''')



        # Основной запрос в GA4 API
        class GA4Request(BaseModelF):
            """Represents a request to the Google Analytics 4 (GA4) Data API."""

            property: constr(pattern=r"^properties/\d+$") = Field(...,
                                                                  description='''The GA4 property identifier in the format 'properties/{property_id}'.
                                                                  For example: 'properties/123456'.''')
            dimensions: Optional[List[Dimension]] = Field(...,
                                                          description='''A list of dimensions to group the data by (e.g., 'country', 'deviceCategory'). 
                                                          Maximum: 9 dimensions.''',
                                                          max_items=9)
            metrics: Optional[List[Metric]] = Field(...,
                                                    description='''A list of metrics to measure the data (e.g., 'newUsers', 'totalRevenue').
                                                    Maximum: 10 metrics.''',
                                                    max_items=10)
            date_ranges: List[DateRange] = Field(...,
                                                 description='''The date range(s) for the report.
                                                 Supports multiple ranges for comparison (e.g., current period vs. previous period).''')
            dimension_filter: Optional[FilterExpression] = Field(None,
                                                                 description="""A filter to include or exclude specific dimension values. 
                                                                 For example, filtering by country or device type. For exclude traffic""")

            metric_filter: Optional[FilterExpression] =  Field(None,
                                                               description="""A filter to include or exclude specific metric values. 
                                                               For example, filtering out revenue below a certain threshold.""")
            order_bys: Optional[List[OrderByItem]] = Field(None,
                                                           description="""A list of sorting rules for the results. 
                                                           For example, sorting by 'totalRevenue' in descending order.""")
            offset: Optional[int] = Field(0,
                                          description="""The starting point for pagination. 
                                          For example, 'offset=100' skips the first 100 rows.""")
            limit: Optional[int] = Field(10000,
                                         description="""The maximum number of rows to return in the response. 
                                         Default is 10,000. Use pagination for larger datasets.""")
            currency_code: Optional[str] = Field(None,
                                                 description="""The currency code for monetary metrics (e.g., 'USD', 'EUR'). 
                                                 If not set, the default currency of the GA4 property is used.""")
            keep_empty_rows: Optional[bool] = Field(False,
                                                    description="""Whether to include rows with no data.
                                                    If True, rows with missing values will still appear in the report.""")
            return_property_quota: Optional[bool] = Field(False,
                                                          description="""Whether to include information about the GA4 property's quota usage in the response.
                                                          Useful for monitoring API limits.""")
            note: Optional[str] = Field(None,
                                        description="""A note about the progress of the work done, information about whether the end result differs for some reason from the user's original request, and a brief description of the API request you created, including its main components: the purpose of the request, the metrics and dimensions used, the time range, and sorting options.""")

            @field_validator("metrics", "dimensions", mode="after")
            @classmethod
            def check_dublicates(cls, metric_list: List):
                for metr in metric_list:
                    if metric_list.count(metr) > 1:
                        raise ValueError(
                            f"InvalidArgument: Duplicate metrics are not allowed. Duplicate metric found: {metr}")
                return metric_list



        class ValidateJsonApiTool(BaseTool):

            name : str = "Google Analytics API request validation tool"
            description: str = ('''This tool is necessary for validating the query data in Google Analytics API, comparing metrics, measurements and other parameters,
                                checking their compatibility and also checking the overall correctness of the JSON query.The format of the validation data in this tool must be GA API JSON, not string.
                                IMPORTANT! Please note that the data passed to the tool must be a valid Python dictionary of keys and values.''')

            args_schema : Type[BaseModel] = GA4Request
            _client_ga: Any = PrivateAttr()

            def __init__(self, client_ga, **kwargs):
                # Сначала инициализируем базу (устанавливаются name, description и т.д.)
                super().__init__(**kwargs)
                # А потом присваиваем приватный атрибут
                self._client_ga = client_ga
            def _run( self, **query) -> Any:

                try:
                    validated_request = GA4Request(**query)  # Проводим валидацию

                except Exception as e:
                    return f"Validation found an Error in structure of your request. Here's some detail about the mistake: \n{e}\nPlease review, fix it, and provide the updated solution."
                try:
                    request = RunReportRequest(
                        property=validated_request.dict().get("property", []),
                        metrics= validated_request.dict().get("metrics", []),  # Преобразуем словари в объекты Metric
                        dimensions= validated_request.dict().get("dimensions", []),
                        # Преобразуем словари в объекты Dimension
                        date_ranges= validated_request.dict().get("date_ranges", []),  # Преобразуем словари в DateRange
                        dimension_filter= validated_request.dict().get("dimension_filter"),
                        metric_filter= validated_request.dict().get("metric_filter"),
                        offset= validated_request.dict().get("offset", 0),  # Значение по умолчанию 0
                        limit= 10,  # Значение по умолчанию 10000
                        metric_aggregations= validated_request.dict().get("metric_aggregations", []),  # Список
                        order_bys= validated_request.dict().get("order_bys", []),  # Список
                        currency_code= validated_request.dict().get("currency_code"),  # Строка
                        cohort_spec= validated_request.dict().get("cohort_spec"),  # Объект или None
                        keep_empty_rows= validated_request.dict().get("keep_empty_rows", False),  # Логическое значение
                        return_property_quota= validated_request.dict().get("return_property_quota", False),
                        # Логическое значение
                        comparisons= validated_request.dict().get("comparisons", []),  # Список
                    )
                    report = self._client_ga.run_report(request)
                    dimension_headers = [header.name for header in report.dimension_headers]
                    metric_headers = [header.name for header in report.metric_headers]
                    rows = []

                    # Извлеките строки данных из ответа
                    for row in report.rows:
                        dimensions = row.dimension_values
                        metrics = row.metric_values
                        row_data = [dim.value for dim in dimensions] + [metric.value for metric in metrics]
                        rows.append(row_data)

                    # Постройте DataFrame с помощью pandas
                    headers = dimension_headers + metric_headers
                    df = pd.DataFrame(rows, columns=headers)


                    return f"This data is well prepared, it has been validated, it is can be used.\n{validated_request.dict()}\nImportant! Make sure that the data you receive with this request matches what the user requested.\nHere is an first 5 rows of the data returned from the query you created.:\n\n{df.to_markdown()}"  # Преобразуем в словарь, если нужно вернуть
                except ValidationError as e:
                    return f"There's an ValidationError in your request. Here's some detail about the mistake: \n{e}\nPlease review, fix it, and provide the updated solution."
                except Exception as e:
                    return f"An Error occurred while trying to send an API request using these parameters. Here's some detail about the mistake: \n{e}\nPlease review, fix it, and provide the updated solution."

        class GA4RequestComparing(BaseModelF):
            overall_verdict: Literal["Fully compliant","Some differences","Total mismatch"] = Field(...,description="Summary judgment on whether the API request fully matches the user's request. Possible values: 'Fully compliant', 'Some differences' or Total mismatch.")
            differences: Optional[str] = Field(None,description="Brief explanation of any discrepancies between the user's request and the generated GA API request. If fully compliant, this should be empty or None.")
            note: Optional[str] = Field(None,description="Additional comments, including any notes left by the specialist who created the API request, providing context for the discrepancies.")

        self.files = {
            'agents': 'at_configs/agent.yaml',
            'tasks': 'at_configs/task.yaml'
        }

        # Load configurations from YAML files
        self.configs = {}
        for config_type, file_path in self.files.items():
            with open(file_path, 'r') as file:
                self.configs[config_type] = yaml.safe_load(file)

        # Assign loaded configurations to specific variables
        agents_config = self.configs['agents']
        tasks_config = self.configs['tasks']
        os.environ['OPENAI_MODEL_NAME'] = ai_model


        self.agent1 = Agent(
            config=agents_config['ga4_api_agent'],
            tools=[],
            memory=True,
        )

        self.task1 = Task(
            config=tasks_config['task_api_generation'],
            agent=self.agent1,
            output_pydantic=GA4Request,
            tools=[ValidateJsonApiTool(client_ga=self.client_ga)],

        )
        self.task2 = Task(
            config=tasks_config['task_api_comparing'],
            agent=self.agent1,
            output_pydantic=GA4RequestComparing,
            context=[self.task1]
        )

        self.crew = Crew(
            agents=[self.agent1],
            tasks=[self.task1],
            verbose=True,
            process = Process.sequential,#hierarchical,manager_llm=ChatOpenAI(temperature=0, model="gpt-4o"),  # Mandatory if manager_agent is not set
            respect_context_window=True,  # Enable respect of the context window for tasks
            memory=True,  # Enable memory usage for enhanced task execution
            manager_agent=None,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
            planning=True,  # Enable planning feature for pre-execution strategy

        )
        self.crew_compare = Crew(
            agents=[self.agent1],
            tasks=[self.task2],
            verbose=True,
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(temperature=0, model="gpt-4o"),  # Mandatory if manager_agent is not set
            respect_context_window=True,  # Enable respect of the context window for tasks
            memory=True,  # Enable memory usage for enhanced task execution
            manager_agent=None,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
            planning=True,  # Enable planning feature for pre-execution strategy

        )





    def create_df_table(self,result):
        dimension_headers = [header.name for header in result.dimension_headers]
        metric_headers = [header.name for header in result.metric_headers]
        rows = []

        # Извлеките строки данных из ответа
        for row in result.rows:
            dimensions = row.dimension_values
            metrics = row.metric_values
            row_data = [dim.value for dim in dimensions] + [metric.value for metric in metrics]
            rows.append(row_data)

        # Постройте DataFrame с помощью pandas
        headers = dimension_headers + metric_headers
        df = pd.DataFrame(rows, columns=headers)
        return df

    def answer(self, user_input):
        result = self.crew.kickoff(inputs={'metrics': self.metrics1,
                                           "dimensions": self.dimensions1,
                                           "user_input": user_input,
                                           "property" : self.ga4_property,
                                           "todays_date" : str(datetime.today().strftime('%Y-%m-%d'))})



        request = RunReportRequest(
            property=f"properties/{self.ga4_property}",
            metrics=result.pydantic.dict().get("metrics", []),  # Преобразуем словари в объекты Metric
            dimensions=result.pydantic.dict().get("dimensions", []),  # Преобразуем словари в объекты Dimension
            date_ranges=result.pydantic.dict().get("date_ranges", []),  # Преобразуем словари в DateRange
            dimension_filter=result.pydantic.dict().get("dimension_filter"),
            metric_filter=result.pydantic.dict().get("metric_filter"),
            offset=result.pydantic.dict().get("offset", 0),  # Значение по умолчанию 0
            limit=result.pydantic.dict().get("limit", 10000),  # Значение по умолчанию 10000
            metric_aggregations=result.pydantic.dict().get("metric_aggregations", []),  # Список
            order_bys=result.pydantic.dict().get("order_bys", []),  # Список
            currency_code=result.pydantic.dict().get("currency_code"),  # Строка
            cohort_spec=result.pydantic.dict().get("cohort_spec"),  # Объект или None
            keep_empty_rows=result.pydantic.dict().get("keep_empty_rows", False),  # Логическое значение
            return_property_quota=result.pydantic.dict().get("return_property_quota", False),  # Логическое значение
            comparisons=result.pydantic.dict().get("comparisons", []),  # Список
        )
        report = self.client_ga.run_report(request)



        # Convert UsageMetrics instance to a DataFrame

        token_dict = result.token_usage.dict()
        # Calculate total costs
        print(f"Total tokenz: {token_dict}")
        costs = (2.50 * token_dict['prompt_tokens'] + 10.0 * token_dict['completion_tokens'])/1000000
        print(f"Total costs: ${costs:.4f}")
            # Display the DataFrame
        return [self.create_df_table(report),result.raw,{key:result.pydantic.dict()[key] for key in ['dimensions','metrics','date_ranges','dimension_filter','metric_filter','limit']}]

    def check_api(self, user_input,result_raw):

        result_compare = self.crew_compare.kickoff(inputs={'metrics': self.metrics1,
                                                           "dimensions": self.dimensions1,
                                                           "property": self.ga4_property,
                                                           "user_input": user_input,
                                                           "json_api": result_raw,
                                                           "todays_date": str(
                                                               datetime.today().strftime('%Y-%m-%d'))})



        # Convert UsageMetrics instance to a DataFrame


        # Display the DataFrame
        return result_compare.pydantic.dict()

