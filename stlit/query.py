import requests
import json
import datetime
import time

def fetch_fitness_data(token, output_file_name="./config/fitdata.json"):
    # Define the endpoint URL
    url = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"

    # Define the JSON data to be sent in the POST request
    xdata = {
      "aggregateBy": [
        {
          "dataTypeName": "com.google.distance.delta",
          "dataSourceId": "derived:com.google.distance.delta:com.google.android.fit:samsung:SM-M536B:10dafbca:top_level"
        },
        {
          "dataTypeName": "com.google.step_count.delta",
          "dataSourceId": "derived:com.google.step_count.delta:com.google.android.fit:samsung:SM-M536B:10dafbca:top_level"
        },
        {
          "dataTypeName": "com.google.calories.expended",
          "dataSourceId": "derived:com.google.calories.expended:com.google.android.gms:merge_calories_expended"
        },
        {
          "dataTypeName": "com.google.heart_rate.bpm",
          "dataSourceId": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
        },
        {
          "dataTypeName": "com.google.sleep.segment",
          "dataSourceId": "derived:com.google.sleep.segment:com.google.android.gms:merged"
        }
      ],
      "bucketByTime": { "durationMillis": 86400000 },
      "startTimeMillis": int(time.time()*1000) - 86400000*2,
      "endTimeMillis": int(time.time()*1000)
    }

    # Define the headers
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json;encoding=utf-8"
    }

    # Make the POST request
    response = requests.post(url, json=xdata, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Request successful!\n")
        # dump with 4 spaces of indentation
        # print(json.dumps(response.json(), indent=4))
        file = response.json()
        # write the data to a JSON file
        with open("./config/response.json", "w") as f:
            json.dump(file, f, indent=4)

    else:
        print("Error:", response.status_code, response.text)

    # print("Data for the last 2 days:\n")

    output = {}

    distance = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.distance.delta:com.google.android.gms:aggregated':
            distance_list = (i['point'][0]['value'])
            for j in distance_list:
                distance.append(j['fpVal'])

    # print("Total distance covered:", sum(distance), "meters")
    output["distance"] = sum(distance)

    steps = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.step_count.delta:com.google.android.gms:aggregated':
            steps_list = (i['point'][0]['value'])
            for j in steps_list:
                steps.append(j['intVal'])

    # print("Total steps taken:", sum(steps))
    output["steps"] = sum(steps)

    calories = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.calories.expended:com.google.android.gms:aggregated':
            calorie_list = (i['point'][0]['value'])
            for j in calorie_list:
                calories.append(j['fpVal'])

    # print("Total calories burned:", sum(calories), "calories")
    output["calories"] = sum(calories)

    heart_rate = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.heart_rate.summary:com.google.android.gms:aggregated':
            heart_rate_list = (i['point'][0]['value'])
            for j in heart_rate_list:
                heart_rate.append(j['fpVal'])

    # print("Average heart rate:", sum(heart_rate)/len(heart_rate), "bpm")
    output["heart_rate"] = sum(heart_rate)/len(heart_rate)

    account_query_url = "https://people.googleapis.com/v1/people/me?personFields=names,emailAddresses"

    # Make the GET request
    response = requests.get(account_query_url, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Request successful!\n")
        # dump with 4 spaces of indentation
        # print(json.dumps(response.json(), indent=4))
        file = response.json()
        # write the data to a JSON file
        with open("./config/account_info.json", "w") as f:
            json.dump(file, f, indent=4)

        name = file["names"][0]["displayName"]
        output["name"] = name
        print("Name:", file["names"][0]["displayName"])

    else:
        print("Error:", response.status_code, response.text)

    with open(output_file_name, "w") as f:
        json.dump(output, f, indent=4)

    return output



def fetch_fitness_data_old(token, output_file_name="./config/fitdata.json"):
    # Define the endpoint URL
    url = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"

    # Define the JSON data to be sent in the POST request
    xdata = {
      "aggregateBy": [
        {
          "dataTypeName": "com.google.distance.delta",
          "dataSourceId": "derived:com.google.distance.delta:com.google.android.fit:samsung:SM-M536B:10dafbca:top_level"
        },
        {
          "dataTypeName": "com.google.step_count.delta",
          "dataSourceId": "derived:com.google.step_count.delta:com.google.android.fit:samsung:SM-M536B:10dafbca:top_level"
        },
        {
          "dataTypeName": "com.google.calories.expended",
          "dataSourceId": "derived:com.google.calories.expended:com.google.android.gms:merge_calories_expended"
        },
        {
          "dataTypeName": "com.google.heart_rate.bpm",
          "dataSourceId": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
        },
        {
          "dataTypeName": "com.google.sleep.segment",
          "dataSourceId": "derived:com.google.sleep.segment:com.google.android.gms:merged"
        }
      ],
      "bucketByTime": { "durationMillis": 86400000 },
      "startTimeMillis": int(time.time()*1000) - 86400000*3,
      "endTimeMillis": int(time.time()*1000)
    }

    # Define the headers
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json;encoding=utf-8"
    }

    # Make the POST request
    response = requests.post(url, json=xdata, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Request successful!\n")
        # dump with 4 spaces of indentation
        # print(json.dumps(response.json(), indent=4))
        file = response.json()
        # write the data to a JSON file
        with open("./config/response.json", "w") as f:
            json.dump(file, f, indent=4)

    else:
        print("Error:", response.status_code, response.text)

    # print("Data for the last 2 days:\n")

    output = {}

    distance = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.distance.delta:com.google.android.gms:aggregated':
            distance_list = (i['point'][0]['value'])
            for j in distance_list:
                distance.append(j['fpVal'])

    # print("Total distance covered:", sum(distance), "meters")
    output["distance"] = sum(distance)

    steps = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.step_count.delta:com.google.android.gms:aggregated':
            steps_list = (i['point'][0]['value'])
            for j in steps_list:
                steps.append(j['intVal'])

    # print("Total steps taken:", sum(steps))
    output["steps"] = sum(steps)

    calories = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.calories.expended:com.google.android.gms:aggregated':
            calorie_list = (i['point'][0]['value'])
            for j in calorie_list:
                calories.append(j['fpVal'])

    # print("Total calories burned:", sum(calories), "calories")
    output["calories"] = sum(calories)

    heart_rate = []
    for i in file['bucket'][0]['dataset']:
        if i['dataSourceId'] == 'derived:com.google.heart_rate.summary:com.google.android.gms:aggregated':
            heart_rate_list = (i['point'][0]['value'])
            for j in heart_rate_list:
                heart_rate.append(j['fpVal'])

    # print("Average heart rate:", sum(heart_rate)/len(heart_rate), "bpm")
    output["heart_rate"] = sum(heart_rate)/len(heart_rate)

    account_query_url = "https://people.googleapis.com/v1/people/me?personFields=names,emailAddresses"

    # Make the GET request
    response = requests.get(account_query_url, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Request successful!\n")
        # dump with 4 spaces of indentation
        # print(json.dumps(response.json(), indent=4))
        file = response.json()
        # write the data to a JSON file
        with open("./config/account_info.json", "w") as f:
            json.dump(file, f, indent=4)

        name = file["names"][0]["displayName"]
        output["name"] = name
        print("Name:", file["names"][0]["displayName"])

    else:
        print("Error:", response.status_code, response.text)

    with open(output_file_name, "w") as f:
        json.dump(output, f, indent=4)

    return output