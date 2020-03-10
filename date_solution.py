import csv
from datetime import datetime, timedelta
from operator import itemgetter

import time
import numpy as np
from cvxopt.glpk import ilp
from cvxopt.modeling import matrix, spmatrix
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
print("Hist ({} items):".format(len(hist)))

hist_preview = PrettyTable(['surrogate_customer_id', 'date', 'channel_code', 'product_code'])

for i in range(3):
    hist_preview.add_row(hist[i])
hist_preview.add_row(['...', '...', '...', '...'])
for i in range(3):
    hist_preview.add_row(hist[len(hist) - i - 1])

print(hist_preview)

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

    if this_date >= start_date:
        break

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
        if communications_channel[this_client][this_channel] is None or communications_channel[this_client][
            this_channel] < matrix_channel[this_channel][i]:
            communications_channel[this_client][i] = matrix_channel[this_channel][i]

shift = 0
while last_date < start_date:
    shift += 1
    last_date += timedelta(days=1)

for k in range(len(communications_channel)):
    for i in range(len(communications_channel[k])):
        communications_channel[k][i] -= shift
        if communications_channel[k][i] < 0:
            communications_channel[k][i] = 0

# теперь начальные ограничения по МКП каналов получены

# print_dates()

# print_communications_channel()

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

model_1d = list3d_to_1d(model)
dates_1d = list3d_to_1d(dates)
c = []
# G = [] # TODO: use a sparse matrix

G_x = []  # values in a sparse matrix
G_i = []  # row indexes in a sparse matrix
G_j = []  # column indexes in a sparse matrix

h = []

A = []
b = []

# the objective function (in order by ijkd: 0000, 0001, 0002, etc...)

for p in range(len(model_1d)):
    for d in range(period_length):
        i, j, k = ijk(model, p)  # i - channel, j - product, k - customer, d - day
        c.append(model_1d[p])
        output.append([p * period_length + d, i, j, k, d, model_1d[p], '?'])

# constraints absolute channel TODO: min range

for i in range(len(channels)):
    # G.append(list())
    # G[i] = list()
    # G[i].extend(list(np.zeros(i * len(products) * len_customers * period_length)))
    # G[i].extend(list(np.ones(len(products) * len_customers * period_length)))
    # G[i].extend(list(np.zeros(period_length * (len(model_1d) - (i + 1) * len(products) * len_customers))))

    G_x.extend(np.ones(len(products) * len_customers * period_length, dtype=float).tolist())
    a = range(i * len(products) * len_customers * period_length,
              (i + 1) * len(products) * len_customers * period_length)
    q = np.ones(len(a), dtype=int) * i
    q = [int(q_item) for q_item in q]

    G_i.extend(q)
    G_j.extend(a)

for constraint in constraint_absolute_channel:
    h.append(constraint[2])

# constraints from history channel

# for k in range(len_customers):
#     for i in range(len(channels)):
#         if communications_channel[k][i] > 0:
#             for j in range(len(products)):
#             #for d in range(communications_channel[k][i]): # зануляем все x по дням от 0 до конца ограничения
#                 # не работает, потому что для n = 7230 communication_channel[k][i] еще не > 0?
#                 n = p3(communications_channel, i, j, k) * 30  # TODO: убрать этот лютый костыль
#                 row = list()
#                 row.extend(list(np.zeros(n)))
#                 row.append(1)
#                 row.extend(list(np.zeros(len_customers * len(channels) * len(products) * period_length - n - 1)))
#                 A.append(row)
#                 b.append(0.0)

# TODO: по каналу ch1 коммуникации идут не чаще чем matrix_channel[ch1][ch2]

# for k in range(len_customers):
#     for j in range(len(products)):
#         for ch1 in range(len(channels)):
#             for ch2 in range(len(channels)):
#                 for d in range(period_length):
#                     row = list(np.zeros(len_customers * len(channels) * len(products) * period_length))
#                     # n = p3(communications_channel, ch1, j, k) * 30
#                     #
#                     # G.append(row)
#                     # h.append(1.0)


# try 1

# G2 = []
# for i in range(len(channels)):
#     G2.append(list())
#     for j in range(len(products)):
#         G2[i].append(list())
#         for k in range(len_customers):
#             G2[i][j].append(list())
#             for d in range(period_length):
#                 G2[i][j][k].append(0.0)
#
# for k in range(len_customers):
#     for j in range(len(products)):
#         for ch in range(len(channels)):
#             for ch2 in range(len(channels)):
#                 if communications_channel[k][ch] < 30:
#                     for d in range(communications_channel[k][ch],
#                                    min(period_length, communications_channel[k][ch] + matrix_channel[ch][ch2])):
#                         G2[ch][j][k][d] = 1.0  # fail in test_id = 1514
#                         # A[ch2][j][k][d] = 1.0  # fail in test_id = 27759, non zeros: 0
#
# h.append(1.0)
#
# G2 = list4d_to_1d(G2)
# G.append(G2)

# try 2

# for k in range(len_customers):
#     for i in range(len(channels)):
#         #row = list(np.zeros(len(channels) * len(products) * len_customers * period_length))
#         # for j in range(len(products)):
#         #     row.append(list())
#         #     for d in range(period_length):
#         #         row[j].append(0.0)
#
#         row = []
#         for i1 in range(len(channels)):
#             row.append(list())
#             for j1 in range(len(products)):
#                 row[i1].append(list())
#                 for k1 in range(len_customers):
#                     row[i1][j1].append(list())
#                     for d in range(period_length):
#                         row[i1][j1][k1].append(0.0)
#
#         for j in range(len(products)):
#             #for ch in range(len(channels)):
#             for ch2 in range(len(channels)):
#                 #if communications_channel_copy[k][i] < period_length and communications_channel_copy[k][i] < matrix_channel[i][ch2]:
#                     #for T in range(communications_channel[k][i], 30-communications_channel[k][i]):
#                 for d in range(0, period_length):
#                     #while d+T < period_length:
#                     for T in range(0, matrix_channel[i][ch2]):
#                             if d + T < period_length:
#                                 row[ch2][j][k][d+T] = 1.0
#                             # row[j][d] = 1.0
#
#         row = list4d_to_1d(row)
#         G.append(row)
#         h.append(1.0)
#         # row = list2d_to_1d(row)

# try 3

eq = G_i[-1] + 1

for k in range(len_customers):
    for i in range(len(channels)):
        for j in range(len(products)):
            for d in range(0, period_length):

                for ch2 in range(len(channels)):
                    row = list()
                    # row = list(np.zeros(len(channels) * len(products) * len_customers * period_length))
                    # while d+T < period_length:
                    for T in range(0, matrix_channel[i][ch2]):
                        if d + T < period_length:
                            # ones from ch2 * len(channels) + j * len(products) + k * len_customers + d to len(row)
                            # row[ch2 * len(channels) + j * len(products) + k * len_customers + d + T] = 1.0
                            row.append(1.0)

                    G_x.extend(row)
                    G_j.extend(range(ch2 * len(channels) + j * len(products) + k * len_customers + d,
                                     ch2 * len(channels) + j * len(products) + k * len_customers + d + len(row)))
                    q = np.ones(len(row)) * eq
                    q = [int(q_item) for q_item in q]

                    G_i.extend(q)
                    h.append(1.0)
                    eq += 1

# print('Press any key to continue solution')
# input()

print(len(G_x))
print(len(G_i))
print(len(G_j))

B = set(range(len(c)))

c = matrix(c)
G = spmatrix(G_x, G_i, G_j)
h = matrix(h)
A = matrix(A).trans()
b = matrix(b)

print('Sizes')
print('c: ', c.size)
print('G: ', G.size)
print('h: ', h.size)
print('A: ', A.size)
print('b: ', b.size)

# TODO: timer

# status, x = ilp(c, G, h, A, b, set(), B)
status, x = ilp(c, G, h, None, None, set(), B)

# Output

table = PrettyTable(['p (Ordinal)', 'Channel', 'Product', 'Client', 'Day', 'Expectation', 'x'])

x = np.array(x)

for i in range(len(output)):
    output[i][6] = float(x[i])

output.sort(key=itemgetter(4))

for p in range(x.size):
    table.add_row(output[p])

output.sort(key=itemgetter(4))

print(table.get_string(start=0, end=5))

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
                                                  constraint_absolute_channel[ch][2] else '\033[31mIncorrect\033[0m'))

print('\n' + 'Step 2: matrix channel')

check2_flag = True
check2_info = ''
non_zeros = 0

for test_id in range(len(output)):
    ch = output[test_id][1]
    prod = output[test_id][2]
    cust = output[test_id][3]
    day = output[test_id][4]

    if day > output[test_id - 1][4]:
        print('Day', day)
        for k in range(len_customers):
            for i in range(len(channels)):
                if communications_channel[k][i] > 0:
                    communications_channel[k][i] -= 1

    this_x = output[test_id][6]
    if this_x != 0.0:
        non_zeros += 1

    if output[test_id][6] == 1.0:
        if communications_channel[cust][ch] > 0:
            check2_flag = False
            check2_info = 'Constraint failed for customer {} at channel {} at product {} at day {}. Channel was forbidden for {} ' \
                          'days more. Additional: test_id = {}, x = {}'.format(cust, ch, prod, day,
                                                                               communications_channel[cust][ch],
                                                                               test_id, output[test_id][6])
            break
        else:
            print('customer {}, {}, {} days to wait. {}. test_id = {}, non zeros: {}'.format(cust, channels[ch],
                                                                                             communications_channel[
                                                                                                 cust][ch],
                                                                                             '\033[32mOK (x={})\033[0m'.format(
                                                                                                 output[test_id][6]),
                                                                                             test_id, non_zeros))
    else:
        print('customer {}, {}, {} days to wait. {}. test_id = {}, non zeros: {}'.format(cust, channels[ch],
                                                                                         communications_channel[cust][
                                                                                             ch],
                                                                                         '\033[32mOK (x={})\033[0m'.format(
                                                                                             output[test_id][6]),
                                                                                         test_id, non_zeros))
        for i in range(len(channels)):
            if matrix_channel[ch][i] > communications_channel[cust][i]:
                communications_channel[cust][i] = matrix_channel[ch][i]

if check2_flag:
    print('Check 2 is submitted.', check2_info)
else:
    print('\033[31mCheck 2 is not submitted.\033[0m', check2_info)

# TODO: check3: matrix product

# TODO: check4: constraint ratio product

objective = 0.0
for item in output:
    objective += -item[5] * item[6]

print("Objective function: {}".format(objective))