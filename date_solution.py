import csv
from datetime import datetime, timedelta
from operator import itemgetter

import numpy as np
from cvxopt.glpk import ilp
from cvxopt.modeling import matrix
from prettytable import PrettyTable

from array_functions import *

# Reading data and types' conversion

with open('model.csv', 'r') as f:
    reader = csv.reader(f)
    dict_model = list(reader)
    for model in dict_model:
        model[0] = int(model[0])

with open('products.csv', 'r') as f:
    reader = csv.reader(f)
    products = list(reader)
    for product in products:
        product[0] = int(product[0])
        product[2] = float(product[2])
    products.sort(key=itemgetter(0))

with open('scores.csv', 'r') as f:
    reader = csv.reader(f)
    scores = list(reader)
    for score in scores:
        score[0] = int(score[0])
        score[1] = int(score[1])
        score[2] = float(score[2])

with open('channels.csv', 'r') as f:
    reader = csv.reader(f)
    channels = list(reader)
    for i in range(0, len(channels)):
        channels[i] = channels[i][0]

with open('hist.csv', 'r') as f:
    reader = csv.reader(f)
    hist = list(reader)
    for item in hist:
        item[0] = int(item[0])
        item[2] = int(item[2])
        item[3] = int(item[3])
        item[1] = datetime.strptime(item[1], '%m/%d/%Y %H:%M')

    # Normalize ids in hist

    last_id = -1
    last_surrogate_id = -1

    hist.sort(key=itemgetter(0))

    for item in hist:
        if item[0] > last_id:
            last_surrogate_id += 1
        last_id = item[0]
        item[0] = last_surrogate_id


with open('matrix_channel.csv', 'r') as f:
    reader = csv.reader(f)
    matrix_channel = []
    for i1 in range(len(channels)):
        matrix_channel.append(list())
        for i2 in range(len(channels)):
            matrix_channel[i1].append(None)

    for line in reader:
        line[0] = channels.index(line[0])
        line[1] = channels.index(line[1])
        line[2] = int(line[2])
        matrix_channel[line[0]][line[1]] = line[2]

with open('matrix_product.csv', 'r') as f:
    reader = csv.reader(f)
    matrix_product = list(reader)
    for item in matrix_product:
        item[2] = int(item[2])

with open('parameters.csv', 'r') as f:
    reader = csv.reader(f)
    parameters = list(reader)
    start_date = parameters[0][0]
    period_length = parameters[0][1]
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    period_length = int(period_length)

with open('constraint_absolute_channel.csv') as f:
    reader = csv.reader(f)
    constraint_absolute_channel = list(reader)

    # limits conversion
    for item in constraint_absolute_channel:
        item[0] = channels.index(item[0])

        if item[1] == '.':
            item[1] = 0
        else:
            item[1] = int(item[1])

        if item[2] == ".":
            item[2] = float("inf")
        else:
            item[2] = int(item[2])

    constraint_absolute_channel.sort(key=itemgetter(0))  # sort by channel_id


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


print("Dict_Model ({} items):".format(len(dict_model)), dict_model[:3], "...")
print("Products ({} items):".format(len(products)), products[:3], "...")
print("Channels ({} items):".format(len(channels)), channels[:3], "...")
print("Scores ({} items):".format(len(scores)), scores[:3], "...")
print('\n')
print("Hist ({} items):".format(len(hist)), hist[:3], "...")
print("Matrix_Channel ({} items):".format(len(matrix_channel)), matrix_channel[:3], "...")
print("Matrix_Product ({} items):".format(len(matrix_product)), matrix_product[:3], "...")
print("Constraint_Absolute_Channel: ({} items):".format(len(constraint_absolute_channel)),
      constraint_absolute_channel[:3], "...")

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

normalized_len = 0
for i in range(len(hist)):
    if hist[i][0] == len_customers - 1 and hist[i + 1][0] == len_customers:
        normalized_len = i

hist = hist[:normalized_len]

# After that we can sort history in chronological order
hist.sort(key=itemgetter(1))

# 4. Creating dates[]

dates = []
for channel in channels:
    dates.append(list())
    for product in products:
        dates[channels.index(channel)].append(list())
        dates[channels.index(channel)][products.index(product)] = list()
        for customer in range(len_customers):
            dates[channels.index(channel)][products.index(product)].append(None)

# 5. Filling dates[]

for item in hist:
    this_client = item[0]
    this_date = item[1]
    this_channel = item[2]
    this_product = item[3] - 1

    if dates[this_channel][this_product][this_client] is None or (
            dates[this_channel][this_product][this_client] < this_date < start_date):
        dates[this_channel][this_product][this_client] = this_date

for i in range(len(dates)):
    for j in range(len(dates[i])):
        for k in range(len(dates[i][j])):
            item = dates[i][j][k]
            if isinstance(item, datetime):
                dates[i][j][k] = int((item.date() - start_date.date()).days)


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


def print_communications_channel():
    print('\nCOMMUNICATIONS CHANNEL')
    for k in range(len(communications_channel)):
        print('customer {}:'.format(k))
        for i in range(len(communications_channel[k])):
            print('\t{}: {} days to wait'.format(channels[i], communications_channel[k][i]))
    return


# Creating and filling communications_channel[]:

communications_channel = []

for k in range(len_customers):
    communications_channel.append(list())
    for i in range(len(channels)):
        communications_channel[k].append(0)

last_date = hist[0][1]

for item in hist:
    this_client = item[0]
    this_date = item[1]
    this_channel = item[2]
    this_product = item[3] - 1

    shift = 0
    while this_date > last_date:
        shift += 1
        last_date += timedelta(days=1)

    for k in range(len(communications_channel)):
        for i in range(len(communications_channel[k])):
            communications_channel[k][i] -= shift
            if communications_channel[k][i] < 0:
                communications_channel[k][i] = 0

    for i in range(len(channels)):
        if communications_channel[this_client][this_channel] is None or communications_channel[this_client][this_channel] < matrix_channel[this_channel][i]:
            communications_channel[this_client][i] = matrix_channel[this_channel][i]

# теперь начальные ограничения по МКП каналов получены

print_dates()

print_communications_channel()

# Solution

# TODO: uncomment optional section when done

# matrix_channel_on = True
# print('Turn off matrix_channel? [Y/n]')
# if input() == 'Y':
#     matrix_channel_on = False
# print('Turn off matrix_product? [Y/n]')

# if matrix_channel_on == False:
#

output = []

model_1d = to_1d(model)
dates_1d = to_1d(dates)
c = []
G = []
h = []

A = []
b = []

for p in range(len(model_1d)):
    # # TODO: try this
    # buf = list(np.ones(3, dtype=float))
    # buf.extend(list(np.zeros(period_length - 3, dtype=float)))
    # G.append(buf)
    # h.append(0.0)

    for d in range(period_length):
        i, j, k = ijk(model, p)  # i - channel, j - product, k - customer, d - day
        c.append(model_1d[p])
        output.append([p * period_length + d, i, j, k, d, model_1d[p], '?'])

# constraints absolute channel

for i in range(len(channels)):
    G.append(list())
    G[i] = list()
    G[i].extend(list(np.zeros(i * len(products) * len_customers * period_length)))
    G[i].extend(list(np.ones(len(products) * len_customers * period_length)))
    G[i].extend(list(np.zeros(period_length * (len(model_1d) - (i + 1) * len(products) * len_customers))))
    # print('rows[{}] ({} items): '.format(i, len(G[i])), G[i])

for constraint in constraint_absolute_channel:
    h.append(constraint[2])

# matrix_channel constraints

B = set(range(len(c)))

c = matrix(c)
G = matrix(G).trans()
h = matrix(h)
A = matrix(A).trans()
b = matrix(b)

print('Sizes')
print('c: ', c.size)
print('G: ', G.size)
print('h: ', h.size)
print('A: ', A.size)
print('b: ', b.size)

status, x = ilp(c, G, h, None, None, set(), B)

# Output

table = PrettyTable(['p (Ordinal)', 'Channel', 'Product', 'Client', 'Day', 'Expectation', 'x'])

x = np.array(x)

for i in range(len(output)):
    output[i][6] = float(x[i])

output.sort(key=itemgetter(4))

for p in range(x.size):
    table.add_row(output[p])

print(table.get_string(start=0, end=31))

# Check constraints

print('\n' + 'CHECKING CONSTRAINTS')

print('Step 1: constraints absolute channel')

check1 = []
for ch in channels:
    check1.append(0)

for item in output:
    this_channel = item[1]
    check1[this_channel] += item[6]

for ch in range(len(channels)):
    print('Total {}: {} ({})'.format(channels[ch], check1[ch],
                                     'Correct' if constraint_absolute_channel[ch][1] <= check1[ch] <=
                                                  constraint_absolute_channel[ch][2] else 'Incorrect'))

print('\n' + 'Step 2: matrix channel')

check2_flag = True

# TODO: переделать полностью
for p in range(len(output)):
    ch = output[p][1]
    cust = output[p][3]
    if x[p] == 1.0:
        if communications_channel[cust][ch] is not None and communications_channel[cust][ch] < 0:
            check2_flag = False


if check2_flag:
    print('Check 2 is submitted.')
else:
    print('Check 2 is not submitted.')

# TODO: check3: matrix product

# TODO: check4: constraint ratio product
