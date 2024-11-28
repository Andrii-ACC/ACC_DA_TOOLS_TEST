import requests
import re
from time import strftime, localtime
import streamlit as st

# Replace with your actual ClickUp API token and Team ID
CLICKUP_API_TOKEN = st.secrets['CLICKUP_API_TOKEN']
CLICKUP_TEAM_ID = st.secrets['CLICKUP_TEAM_ID']

headers = {
    'Authorization': CLICKUP_API_TOKEN
}


def get_ab_test_data_by_client_name(client_name):
    # Get all spaces in the team
    spaces_url = f'https://api.clickup.com/api/v2/team/{CLICKUP_TEAM_ID}/space'
    spaces_response = requests.get(spaces_url, headers=headers).json()

    for space in spaces_response['spaces']:
        # Get all folders in the space
        folders_url = f'https://api.clickup.com/api/v2/space/{space["id"]}/folder'
        folders_response = requests.get(folders_url, headers=headers).json()

        for folder in folders_response['folders']:

            if folder['name'] == client_name:
                # Found the folder, now get the lists
                lists_url = f'https://api.clickup.com/api/v2/folder/{folder["id"]}/list'
                lists_response = requests.get(lists_url, headers=headers).json()
                for lst in lists_response['lists']:

                    if lst['name'] == 'A/B Test Tasks':
                        # Found the AB Testing list, now get the tasks (tickets)
                        tasks_url = f'https://api.clickup.com/api/v2/list/{lst["id"]}/task?include_subtasks=true'
                        tasks_response = requests.get(tasks_url, headers=headers).json()
                        ab_tests = []

                        for task in tasks_response['tasks']:

                            task["simplified user fields"] = {}
                            if "Test Hypothesis" in task['text_content']:

                                custom_fields = task["custom_fields"]

                                for custom_field in custom_fields:
                                    if "value" in custom_field.keys():
                                        value = custom_field['value']
                                        name = custom_field['name']
                                        if "options" in custom_field["type_config"].keys() and len(custom_field["type_config"]["options"]) !=0:
                                            for option in custom_field["type_config"]["options"]:
                                                if "orderindex" in option.keys() and option["orderindex"] == value:
                                                    value = option["name"]
                                                    break
                                        task["simplified user fields"][name] = value




                                context_of_hypothesis = re.search(r"IF.*\nTHEN.*\nBECAUSE.*\n", task['text_content'])
                                if context_of_hypothesis:
                                    context_of_hypothesis = context_of_hypothesis.group(0)
                                else:
                                    context_of_hypothesis = "A/B test exists, but hypothesis not found."
                                ab_tests.append({
                                    'id' : task['id'],
                                    'name': task['name'],
                                    'date created' : strftime('%Y-%m-%d', localtime(int(task['date_created'][:-3]))),
                                    'date updated' : strftime('%Y-%m-%d', localtime(int(task['date_updated'][:-3]))),
                                    'context of hypothesis' : context_of_hypothesis,
                                    'status': task['status']['status'],
                                    'simplified user fields' : task["simplified user fields"]

                                })
                            # Check if the task is in the 'AB Testing' subcategory
                            # Assuming subcategory is a custom field or tag
                            # Modify the following condition based on your actual setup
                        return ab_tests
    return None

# Usage example
folder_name_input = 'Educate'
def get_ab_test_tickets_info_by_client_name (client_name):
    list_of_tests = get_ab_test_data_by_client_name(folder_name_input)
    finall_prompt_list = []
    for test in list_of_tests:
        if test['simplified user fields']['Task Stage'] == "Killed":
            outcome = "Conversion rates have not increased or the increase has been insignificant"
        elif test['simplified user fields']['Task Stage'] in ["Deploy Live", "Live"]:
            outcome = "The conversion rate increased enough to change the original to the tested variant."
        else:
            continue




        add_prompt = f"""
    **Test Name**: {test["name"]}
    
  - **Hypothesis**: {test["context of hypothesis"]}
  - **Result**: {outcome}
  - **Parameters**:
    - *Test creation date*: {test['date created']}
    - *Test completion date*: {test['date updated']}"""
        finall_prompt_list.append(add_prompt)
    finall_prompt ='\n\n'.join(finall_prompt_list)
    finall_prompt = "\nBelow is information on the A/B tests already conducted for this client:\n\n"+finall_prompt
    return finall_prompt

#ab_test_results = get_ab_test_tickets_info_by_client_name(folder_name_input)
def get_list_of_clients_names():
    print("GETING LIST!")
    spaces_url = f'https://api.clickup.com/api/v2/team/{CLICKUP_TEAM_ID}/space'
    spaces_response = requests.get(spaces_url, headers=headers).json()
    clients_folders_names_list = []
    for space in spaces_response['spaces']:

        if re.fullmatch(r".*team$", space['name'].lower()):
            folders_url = f'https://api.clickup.com/api/v2/space/{space["id"]}/folder'
            folders_response = requests.get(folders_url, headers=headers).json()

            for folder in folders_response['folders']:
                clients_folders_names_list.append(folder['name'])
    return clients_folders_names_list
