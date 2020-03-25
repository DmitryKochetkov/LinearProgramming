import csv
from datetime import datetime, timedelta
from random import randint, triangular, randrange, random, seed

from array_functions import *
from operator import itemgetter

channels = 0
products = 0

seed()

with open('channels.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        channels += 1

with open('products.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        products += 1

with open('parameters.csv', 'r') as f:
    reader = csv.reader(f)
    parameters = list(reader)
    start_date = parameters[0][0]
    period_length = parameters[0][1]
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    period_length = int(period_length)

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
            if scores[i][0] == desired_clients:
                scores = scores[:i-1]
                print('Scores truncated')
                break

    #  если клиентов в scores меньше чем требуется, то генерируем клиентов с случайными model и вероятностью
    if len_customers < desired_clients:
        for k in range(len_customers, desired_clients):
            for m in range(len(dict_model)):
                if randint(0, 10) > 7:
                    scores.append([k, m, round(triangular(min_probability, max_probability), 5)])
                    print('Scores: добавлена запись [cust_id {}, model {}, probability = {}]'.format(scores[-1][0], scores[-1][1], scores[-1][2]))

with open('scores_generated.csv', "w") as f:
    writer = csv.writer(f)
    f.truncate()
    writer.writerows(scores)

creation_of_the_world = start_date

with open('hist.csv', 'r') as f:
    reader = csv.reader(f)
    hist = list(reader)
    for item in hist:
        item[0] = int(item[0])
        item[2] = int(item[2])
        item[3] = int(item[3])
        item[1] = datetime.strptime(item[1], '%m/%d/%Y %H:%M')
        if item[1] < creation_of_the_world:
            creation_of_the_world = item[1]

        # ID из истории нормализуются, т.е. сопоставляются с ID в dict_model

    last_id = -1
    last_surrogate_id = -1

    hist.sort(key=itemgetter(0))

    for item in hist:
        if item[0] > last_id:
            last_surrogate_id += 1
        last_id = item[0]
        item[0] = last_surrogate_id

    if last_surrogate_id > desired_clients:
        for i in range(len(hist)):
            if hist[i][0] == desired_clients:
                hist = hist[:i-1]
                print('Hist truncated')
                break

    if last_surrogate_id < desired_clients:
        for k in range(desired_clients - last_surrogate_id, desired_clients):
            for n in range(randint(3, 8)): # генерируем сколько-нибудь коммуникаций от 3 до 8
                d = randrange((start_date - creation_of_the_world).days - 1) # в дату от "сотворения мира" до начала оптимизации
                d = start_date + timedelta(days=d, seconds=1)
                if randint(0, 1) == 1:
                    hist.append([k, d, randint(0, channels - 1), randint(0, products - 1)]) # случайную коммуникацию
                    print('Hist: добавлена запись [cust_id {}, day {}, channel {}, product {}]'.format(hist[-1][0], hist[-1][1], hist[-1][2], hist[-1][3]))

for item in hist:
    item[1] = item[1].strftime('%m/%d/%Y %H:%M')

with open('hist_generated.csv', "w") as f:
    writer = csv.writer(f)
    f.truncate()
    writer.writerows(hist)