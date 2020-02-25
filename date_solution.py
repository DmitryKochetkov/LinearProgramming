import csv
from cvxopt.modeling import variable, op, matrix, sum, _function
from cvxopt.glpk import ilp
import numpy as np
from operator import itemgetter
import operator
from prettytable import PrettyTable
from datetime import datetime

# Reading data
with open('model.csv', 'r') as f:
    reader = csv.reader(f)
    dict_model = list(reader)

with open('products.csv', 'r') as f:
    reader = csv.reader(f)
    products = list(reader)

with open('scores.csv', 'r') as f:
    reader = csv.reader(f)
    scores = list(reader)

with open('channels.csv', 'r') as f:
    reader = csv.reader(f)
    channels = list(reader)

with open('hist.csv', 'r') as f:
    reader = csv.reader(f)
    hist = list(reader)

with open('matrix_channel.csv', 'r') as f:
    reader = csv.reader(f)
    matrix_channel = list(reader)

with open('matrix_product.csv', 'r') as f:
    reader = csv.reader(f)
    matrix_product = list(reader)

with open('parameters.csv', 'r') as f:
    reader = csv.reader(f)
    parameters = list(reader)
    start_date = parameters[0][0]
    period_length = parameters[0][1]
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    period_length = int(period_length)


# Useful functions for dictionary_model conversion

def profit_by_model(model_code):
    m = dict_model[model_code]
    product_name = m[2]
    for product in products:
        if product[1] == product_name:
            return product[0]
    return 0


def channel_by_model(model_code):
    for m in dict_model:
        if m[0] == model_code:
            return channels.index(m[3])  # m3 is a channel in dict_model[i]

    return -1


def product_by_model(model_code):
    for m in dict_model:
        if m[0] == model_code:
            for p in products:
                if p[1] == m[2]:
                    return p[0] - 1  # requires products to be sorted by id
    return -2


# Data types conversion

for model in dict_model:
    model[0] = int(model[0])

for product in products:
    product[0] = int(product[0])
    product[2] = float(product[2])

for item in hist:
    item[0] = int(item[0])
    item[2] = int(item[2])
    item[3] = int(item[3])
    item[1] = datetime.strptime(item[1], '%m/%d/%Y %H:%M')

for item in matrix_product:
    item[2] = int(item[2])

for item in matrix_channel:
    item[2] = int(item[2])

products.sort(key=itemgetter(0))

for i in range(0, len(channels)):
    channels[i] = channels[i][0]

# scores = scores[:15]  # Simplify

for score in scores:
    score[0] = int(score[0])
    score[1] = int(score[1])
    score[2] = float(score[2])

hist.sort(key=itemgetter(0))

# Normalize ids in hist

last_id = -1
last_surrogate_id = -1

for item in hist:
    if item[0] > last_id:
        last_surrogate_id += 1
    last_id = item[0]
    item[0] = last_surrogate_id

print("Dict_Model ({} items):".format(len(dict_model)), dict_model[:3], "...")
print("Products ({} items):".format(len(products)), products[:3], "...")
print("Channels ({} items):".format(len(channels)), channels[:3], "...")
print("Scores ({} items):".format(len(scores)), scores[:3], "...")
print('\n')
print("Hist ({} items):".format(len(hist)), hist[:3], "...")
print("Matrix_Channel ({} items):".format(len(matrix_channel)), matrix_channel[:3], "...")
print("Matrix_Product ({} items):".format(len(matrix_product)), matrix_product[:3], "...")
print("Start Date:", start_date)
print("Period Length: {} days".format(period_length))

# Preparing data

# 0. Count customers (for auto increments users' ids only)
len_customers = -1
for score in scores:
    if score[0] > len_customers:
        len_customers += 1

len_customers += 1

print('\n' + 'CHECKING CLIENTS')
clients_scores = set()
clients_hist = set()
for item in scores:
    clients_scores.add(item[0])

for item in hist:
    clients_hist.add(item[0])

print('Total clients in Scores:', len(clients_scores))
print('Total clients in Hist:', len(clients_hist))
if len(clients_scores.intersection(clients_hist)) == 0:
    print('Warning: intersection is an empty set')


# 1. Creating model[]
model = []

for channel in channels:
    model.append(list())
    for product in products:
        model[channels.index(channel)].append(list())
        model[channels.index(channel)][products.index(product)] = list()
        for customer in range(len_customers):
            model[channels.index(channel)][products.index(product)].append(0)

# 2. Filling model
for score in scores:
    this_client = score[0]
    this_model = score[1]
    this_probability = score[2]

    this_channel = channel_by_model(this_model)
    this_product = product_by_model(this_model)

    model[this_channel][this_product][this_client] = -this_probability * products[this_product][2]

# 3. Cut extra data from hist

print("Normalized hist:")

normalized_len = 0
for i in range(len(hist)):
    if hist[i][0] == len_customers - 1 and hist[i+1][0] == len_customers:
        normalized_len = i

hist = hist[:normalized_len]

for item in hist:
    print(item)

# 4. Creating dates[]

dates = []
for channel in channels:
    dates.append(list())
    for product in products:
        dates[channels.index(channel)].append(list())
        dates[channels.index(channel)][products.index(product)] = list()
        for customer in range(len_customers):
            dates[channels.index(channel)][products.index(product)].append(0)

# 5. Filling dates[]

for item in hist:
    this_client = item[0]
    this_date = item[1]
    this_channel = item[2]
    this_product = item[3] - 1

    if dates[this_channel][this_product][this_client] == 0 or (this_date > dates[this_channel][this_product][this_client] and this_date < start_date):
        dates[this_channel][this_product][this_client] = this_date

# Model[] and Dates[] print functions

def print_model():
    print('\nMODEL')
    for i in range(len(model)):
        print('{}:'.format(channels[i]))
        for j in range(len(model[i])):
            print('\t{}:'.format(products[j]))
            for k in range(len(model[i][j])):
                print('\t\tcustomer {}: {}'.format(k, model[i][j][k]))
    return

def print_dates():
    print('\nDATES')
    for i in range(len(dates)):
        print('{}:'.format(channels[i]))
        for j in range(len(dates[i])):
            print('\t{}:'.format(products[j]))
            for k in range(len(dates[i][j])):
                print('\t\tcustomer {}: {}'.format(k, dates[i][j][k]))
    return

print_dates()

# Array functions TODO: move into separate module

def to_1d(arr3d):
    arr1d = []
    for i in range(len(arr3d)):
        for j in range(len(arr3d[i])):
            for k in range(len(arr3d[i][j])):
                arr1d.append(arr3d[i][j][k])
    return arr1d


def ijk(arr_3d, p):  # предполагается матрица (т.е массив, у которого длина подмассивов на всех уровнях одинакова)
    i = p // len(arr_3d[0][0]) // len(arr_3d[0])
    j = p // len(arr_3d[0][0]) % len(arr_3d[0])
    k = p % len(arr_3d[0][0])
    return (i, j, k)


def a_ijk(arr3d, p):
    i, j, k = ijk(arr3d, p)
    return arr3d[i][j][k]


# Solution

model_1d = to_1d(model)
c = matrix(model_1d)

rows = []

for i in range(len(channels)):
    rows.append(list())
    rows[i] = list()
    rows[i].extend(list(np.zeros(i * len(products) * len_customers)))
    rows[i].extend(list(np.ones(len(products) * len_customers)))
    rows[i].extend(list(np.zeros(len(model_1d) - (i + 1) * len(products) * len_customers)))

G = matrix(rows).trans()

h = matrix(np.array([10, 50, 100, 20], dtype=float))

B = set(range(len(model_1d)))

# status, x = ilp(c, G, h, None, None, set(), B)

# Output

# table = PrettyTable(['p (Ordinal)', 'Channel', 'Product', 'Client', 'Expectation', 'x'])
#
# for p in range(len(model_1d)):
#     i, j, k = ijk(model, p)
#     table.add_row([p, i, j, k, c[p], x[p]])

# print(table.get_string(sort_key=operator.itemgetter(5, 4), sortby="Expectation")) # just for me to explore, doesn't work btw
# print(table)

# Check constraints

print('\n' + 'CHECKING CONSTRAINTS')

# check = []
# for ch in channels:
#     check.append(0)
#
# for p in range(len(model_1d)):
#     this_channel = ijk(model, p)[0]
#     check[this_channel] += x[p]
#
# for ch in range(len(channels)):
#     print('Total {}: {}'.format(channels[ch], check[ch]))
