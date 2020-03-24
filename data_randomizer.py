import csv
from datetime import datetime, timedelta
from random import randint, triangular

channels = 0
products = 0

with open('channels.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        channels += 1

with open('products.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        products += 1

with open('model.csv', 'r') as f:
    reader = csv.reader(f)
    dict_model = list(reader)
    for model in dict_model:
        model[0] = int(model[0])

print('Введите желаемое количество клиентов: ')
desired_clients = int(input())

min_probability = 1.0
max_probability = 0.0

with open('scores.csv', 'r') as f:
    reader = csv.reader(f)
    scores = list(reader)
    for score in scores:
        score[0] = int(score[0])
        score[1] = int(score[1])
        score[2] = float(score[2])

        if score[2] > max_probability:
            max_probability = score[2]

        if score[2] < min_probability:
            min_probability = score[2]

    # Считаем, сколько различных id клиентов в scores
    len_customers = -1
    for score in scores:
        if score[0] > len_customers:
            len_customers += 1

    len_customers += 1

    if len_customers > desired_clients:
        for i in range(len(scores)):
            if score[i][0] > desired_clients:
                scores = scores[:i-1]
                break

    #  если клиентов в scores меньше чем требуется, то генерируем клиентов с случайными model и вероятностью
    if len_customers < desired_clients:
        for k in range(desired_clients - len_customers, desired_clients):
            for m in range(len(dict_model)):
                if randint(0, 10) > 7:
                    scores.append([k, m, round(triangular(min_probability, max_probability), 5)])
                    print('Scores: добавлена запись [cust_id {}, model {}, probability = {}]'.format(scores[-1][0], scores[-1][1], scores[-1][2]))

# with open('scores_generated.csv', "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(scores)

with open('hist.csv', 'r') as f:
    reader = csv.reader(f)
    hist = list(reader)
    for item in hist:
        item[0] = int(item[0])
        item[2] = int(item[2])
        item[3] = int(item[3])
        item[1] = datetime.strptime(item[1], '%m/%d/%Y %H:%M')