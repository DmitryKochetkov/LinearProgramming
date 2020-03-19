import csv
from datetime import datetime, timedelta
from operator import itemgetter

import numpy as np
from cvxopt.glpk import ilp
from cvxopt.modeling import matrix, spmatrix
from prettytable import PrettyTable

from array_functions import *

# Чтение данных из csv с конвертацией в нужные типы

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

    # ID из истории нормализуются, т.е. сопоставляются с ID в dict_model

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

# Параметры оптимизации - стартовая дата и период
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

# 1. Создаем model как трехмерный список (канал-продукт-клиент)
model = []

for channel in channels:
    model.append(list())
    for product in products:
        model[channels.index(channel)].append(list())
        model[channels.index(channel)][products.index(product)] = list()
        for customer in range(len_customers):
            model[channels.index(channel)][products.index(product)].append(0)

# 2. Заполняем model данными из таблицы
for score in scores:
    this_client = score[0]
    this_model = score[1]
    this_probability = score[2]

    this_channel = channel_by_model(this_model)
    this_product = product_by_model(this_model)

    model[this_channel][this_product][this_client] = -this_probability * products[this_product][2]

# 3. Удаляем из истории данные о клиентах, отсутствующих в модели (часть нормализации ID клиентов)

normalized_len = 0
for i in range(len(hist)):
    if hist[i][0] == len_customers - 1 and hist[i + 1][0] == len_customers:
        normalized_len = i

hist = hist[:normalized_len]

# Сортируем истории в хронологическом порядке (по датам)
hist.sort(key=itemgetter(1))


# print functions

def print_model():
    print('\nMODEL')
    for i in range(len(model)):
        print('{}:'.format(channels[i]))
        for j in range(len(model[i])):
            print('\t{}:'.format(products[j]))
            for k in range(len(model[i][j])):
                print('\t\tcustomer {}: {}'.format(k, model[i][j][k]))
    return


def print_communications_channel(cust=None):
    print('\nCOMMUNICATIONS CHANNEL', 'for customer {}'.format(cust) if cust is not None else '')
    if isinstance(cust, int):
        print('customer {}:'.format(cust))
        for i in range(len(communications_channel[cust])):
            print('\t{}: {} days to wait'.format(channels[i], communications_channel[cust][i]))
    else:
        for k in range(len(communications_channel)):
            print('customer {}:'.format(k))
            for i in range(len(communications_channel[k])):
                print('\t{}: {} days to wait'.format(channels[i], communications_channel[k][i]))


# communications_channel будет содержать данные об ограничениях из истории до дня начала оптимизации
# Суть: Коммуникации с клиентом по каналу запрещены еще communication_channel[клиент][канал] дней от начала оптимизации
# После решения этот список перезаписывается в соответствии с корнями для проверки их корректности
# TODO: для реализации ограничений по МКП по продуктам потребуется аналогичный список communication_product[клиент][продукт]

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

# теперь получены cтартовые ограничения из истории (пока только для МКП по каналам)

# РЕШЕНИЕ

# Здесь прототип фичи для возможности выбора ограничений пользователем
# TODO: раскомментировать, когда все будет готово

# matrix_channel_on = True
# print('Turn off matrix_channel? [Y/n]')
# if input() == 'Y':
#     matrix_channel_on = False
# print('Turn off matrix_product? [Y/n]')

# if matrix_channel_on == False:
#

output = []  # отдельный двумерный список для красивого вывода, каждая его строка помещается в красивую табличку

# неравенство-ограничение имеет вид G*x <= h, уравнение-ограничение имеет вид A*x <= b
model_1d = list3d_to_1d(model)
c = []  # коэффициенты при x в целевой функции

# G станет разреженной матрицей коэффициентов неравенства-ограничения
G_x = []  # значения в разреженной матрице
G_i = []  # индексы строк для значений в разреженной матрице
G_j = []  # индексы столбцов для значений в разреженной матрице

h = []  # матрица свободных членов неравенства-ограничения

A = []  # матрица коэффициентов уравнения-ограничения

b = []  # матрица свободных членов уравнения-ограничения

# целевая функция (порядок иксов ijkd: 0000, 0001, 0002, ..., 000_30, 001_0 etc...)

for p in range(len(model_1d)):
    for d in range(period_length):
        i, j, k = ijk(model, p)  # i - channel, j - product, k - customer, d - day
        c.append(model_1d[p])
        output.append([p * period_length + d, i, j, k, d, model_1d[p], '?'])

# ограничения на общее количество коммуникаций по каналу TODO: решить, что делать с нижней границей

for i in range(len(channels)):
    G_x.extend(np.ones(len(products) * len_customers * period_length, dtype=float).tolist())
    a = range(i * len(products) * len_customers * period_length,
              (i + 1) * len(products) * len_customers * period_length)
    q = np.ones(len(a), dtype=int) * i
    q = [int(q_item) for q_item in q]

    G_i.extend(q)
    G_j.extend(a)

for constraint in constraint_absolute_channel:
    h.append(constraint[2])

# ограничения из истории

A = np.zeros(len(channels) * len(products) * len_customers * period_length, dtype=float).tolist()
for i in range(len(channels)):
    for j in range(len(products)):
        for k in range(len_customers):
            for d in range(period_length):
                index = i * len(
                    products) * len_customers * period_length + j * len_customers * period_length + k * period_length + d
                if d < communications_channel[k][i]:
                    A[index] = 1.0

b = [0.0]

# ограничения из МКП по каналам

neq = G_i[-1] + 1  # номер последней строки матрицы коэффициентов неравенства

# for k in range(len_customers):
#     for d in range(period_length):
#
#         for i in range(len(channels)):
#             inequality_x = list()
#             inequality_i = list()
#             inequality_j = list()
#
#             for j in range(len(products)):
#                 for ch2 in range(len(channels)):
#                     for T in range(min(matrix_channel[i][ch2], period_length)):
#                         if d + T < period_length:
#                             index = ch2 * len(
#                                 products) * len_customers * period_length + j * len_customers * period_length + k * period_length + d + T
#                             inequality_x.append(1.0)
#                             inequality_i.append(neq)
#                             inequality_j.append(index)
#
#             G_x.extend(inequality_x)
#             G_i.extend(inequality_i)
#             G_j.extend(inequality_j)
#             h.append(1.0)
#             neq += 1

# FIRST SUCCESSFUL TRY
# for k in range(len_customers):
#     for d in range(period_length):
#
#         inequality_x = list()
#         inequality_i = list()
#         inequality_j = list()
#
#         for i in range(len(channels)):
#             T_max = 0
#             for ch2 in range(len(channels)):
#                 if matrix_channel[i][ch2] > T_max:
#                     T_max = matrix_channel[i][ch2]
#
#             for j in range(len(products)):
#                 for T in range(min(T_max, period_length)):
#                     if d + T < period_length:
#                         index = i * len(
#                             products) * len_customers * period_length + j * len_customers * period_length + k * period_length + d + T
#                         inequality_x.append(1.0)
#                         inequality_i.append(neq)
#                         inequality_j.append(index)
#
#         G_x.extend(inequality_x)
#         G_i.extend(inequality_i)
#         G_j.extend(inequality_j)
#         h.append(1.0)
#         neq += 1

for k in range(len_customers):
    for d in range(period_length):

        inequality_x = list()
        inequality_i = list()
        inequality_j = list()

        for i in range(len(channels)):
            T_max = 0
            for ch2 in range(len(channels)):
                if matrix_channel[i][ch2] > T_max:
                    T_max = matrix_channel[i][ch2]

            for j in range(len(products)):
                for T in range(min(T_max, period_length)):
                    if d + T < period_length:
                        index = i * len(
                            products) * len_customers * period_length + j * len_customers * period_length + k * period_length + d + T
                        inequality_x.append(1.0)
                        inequality_i.append(neq)
                        inequality_j.append(index)

        G_x.extend(inequality_x)
        G_i.extend(inequality_i)
        G_j.extend(inequality_j)
        h.append(1.0)
        neq += 1

print(len(G_x))
print(len(G_i))
print(len(G_j))

B = set(range(len(c)))

c = matrix(c)
G = spmatrix(G_x, G_i, G_j)
h = matrix(h)
A = matrix(A).trans()
b = matrix(b)

# TODO: попробовать уменьшить количество неравенств, подаваемых на вход оптимизатору
# amount = 500
# G.
# G = G[:amount] # придумать способ с разреженной матрицей
# h = h[:amount]

print('Sizes')
print('c: ', c.size)
print('G: ', G.size)
print('h: ', h.size)
print('A: ', A.size)
print('b: ', b.size)

status, x = ilp(c, G, h, A, b, set(), B)

# Output

table = PrettyTable(['p (Ordinal)', 'Channel', 'Product', 'Client', 'Day', 'Expectation', 'x'])

x = np.array(x)

# копируем иксы в нужную колонку списка output (это все сделано только для красивой инициализации PrettyTable)
# наверняка это можно сделать адекватнее

for i in range(len(output)):
    output[i][6] = float(x[i])

output.sort(key=itemgetter(4))  # сортируем вывод по датам

for p in range(x.size):
    table.add_row(output[p])

# Проверка правильности всех ограничений

print('\n' + 'CHECKING CONSTRAINTS')


def ask(q='Start?'):
    print(q, '[Y/n]')
    response = input()
    return response == 'Y'


print('\n' + 'Step 1: constraints absolute channel')
if ask():
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

print('\n' + 'Step 2: constraints absolute channel')
if ask():
    check2 = []
    for ch in channels:
        check2.append(0)

    for item in output:
        this_channel = item[1]
        check2[this_channel] += item[6]

    for ch in range(len(channels)):
        print('Total {}: {} ({})'.format(channels[ch], check2[ch],
                                         'Correct' if constraint_absolute_channel[ch][1] <= check2[ch] <=
                                                      constraint_absolute_channel[ch][2] else '\033[31mIncorrect\033[0m'))

print('\n' + 'Step 3: matrix channel')
if ask():
    check3_flag = True
    check3_info = ''  # информация о том, как прошло тестирование (вернее, на каком корне оно не прошло)
    non_zeros = 0  # количество ненулевых корней на текущий момент тестирования (чтобы удостовериться, что не все иксы занулились)

    opt_hist = []  # история, дополненная коммуникациями на период оптимизации

    # даты до оптимизации рассчитываются относительно начала оптимизации (день -3 это дата за три дня до начала оптимизации)
    for item in hist:
        if item[1].day - start_date.day < 0:
            opt_hist.append([item[0], item[1].day - start_date.day, item[2], item[3]])

    print_communications_channel(cust=19)

    # моделируем процесс коммуникаций по найденным решениям
    for test_id in range(len(output)):
        # получаем канал, продукт, клиент, день и найденный корень для текущей коммуникации
        ch = output[test_id][1]
        prod = output[test_id][2]
        cust = output[test_id][3]
        day = output[test_id][4]
        this_x = output[test_id][6]

        # Если настал новый день, то все ограничения, лежавшие в communications_channel, уменьшаются на единицу
        if day > output[test_id - 1][4]:
            print('Day', day)
            for k in range(len_customers):
                for i in range(len(channels)):
                    if communications_channel[k][i] > 0:
                        communications_channel[k][i] -= 1

        if this_x != 0.0:
            non_zeros += 1

        if this_x == 1.0:
            # Если коммуникация произошла вопреки ограничению, то тест не пройден.
            if communications_channel[cust][ch] > 0:
                check3_flag = False
                check3_info = 'Constraint failed for customer {} at channel {} at product {} at day {}. Channel was forbidden for {} ' \
                              'days more. Additional: communication_id = {}, x = {}, p = {}'.format(cust, ch, prod, day,
                                                                                                    communications_channel[
                                                                                                        cust][
                                                                                                        ch],
                                                                                                    test_id,
                                                                                                    output[test_id][6],
                                                                                                    output[test_id][0])
                print_communications_channel(cust=cust)
                opt_hist.sort(key=itemgetter(1))
                opt_hist.reverse()

                # дальше какая-то тупая проверка индексов, я в них уже запутался, все это нужно будет выкинуть отсюда

                index = ch * len(
                    products) * len_customers * period_length + prod * len_customers * period_length + cust * period_length + day
                print('index for this test is', index)
                print('A[index] = {}'.format(A[index]))

                # ищем последнюю коммуникацию с данным клиентом, чтобы понять, насколько все плохо

                for item in opt_hist:
                    if item[0] == cust:
                        check3_info += '\nThe last communication with customer {} was on day {} (channel {}, product {})'.format(
                            cust, item[1], item[2], item[3])
                        # TODO: для отладки нужна не только последняя коммуникация с клиентом, она нужна еще и по тому же каналу
                        break
                print()

                #  Выводим все неравенства, содержащие переменную, не прошедшую тест
                inequalities = dict()

                # проходимся по неравенствам и запоминаем, какие из них содержат переменные, которые хотим посмотреть
                for n in range(len(G_i)):
                    i = G_j[n] // period_length // len_customers // len(products)
                    j = G_j[n] // period_length // len_customers % len(products)
                    k = G_j[n] // period_length % len_customers
                    d = G_j[n] % period_length

                    if i == ch and j == prod and k == cust and d == day:
                        # inequalities[inequality_id] = ('x_i{}_j{}_k{}_d{} + '.format(i, j, k, d))
                        inequalities[G_i[n]] = list()

                # проходимся по переменным, и смотрим есть ли она в неравенстве, которое хотим посмотреть
                for inequality_item in range(len(G_j)):
                    if G_i[inequality_item] in inequalities.keys():
                        inequalities[G_i[inequality_item]].append(G_j[inequality_item])

                # восстанавливаем четырехмерность каждого индекса
                for value in inequalities.values():
                    for n in range(len(value)):
                        i = value[n] // period_length // len_customers // len(products)
                        j = value[n] // period_length // len_customers % len(products)
                        k = value[n] // period_length % len_customers
                        d = value[n] % period_length
                        value[n] = 'x_i{}j{}k{}d{}'.format(i, j, k, d)

                # вывод неравенств
                print('Total inequalities: {}.'.format(len(inequalities)))
                for k, v in inequalities.items():
                    print('Inequality {} ({} слагаемых): {} {}'.format(k, len(v), v,
                                                                       'Empty inequality' if len(v) == 0 else '< 1'))
                # TODO: вывести уравнения

                break
            else:
                print(
                    'customer {}, {}, {} days to wait. {}. communication_id = {}, non zeros: {}'.format(cust, channels[ch],
                                                                                                        communications_channel[
                                                                                                            cust][ch],
                                                                                                        '\033[32mOK (x={})\033[0m'.format(
                                                                                                            output[test_id][
                                                                                                                6]),
                                                                                                        test_id, non_zeros))

                # Если же ограничения не было и коммуникация произошла, добавляем ее в историю оптимизации...
                opt_hist.append([cust, day, ch, prod])
                # ... и ставим новое ограничение из МКП
                for i in range(len(channels)):
                    if matrix_channel[ch][i] > communications_channel[cust][i]:
                        communications_channel[cust][i] = matrix_channel[ch][i]

        else:
            print('customer {}, {}, {} days to wait. {}. communication_id = {}, non zeros: {}'.format(cust, channels[ch],
                                                                                                      communications_channel[
                                                                                                          cust][
                                                                                                          ch],
                                                                                                      '\033[32mOK (x={})\033[0m'.format(
                                                                                                          output[test_id][
                                                                                                              6]),
                                                                                                      test_id, non_zeros))

    if check3_flag:
        print('Check 2 is submitted.', check3_info)
    else:
        print('\033[31mCheck 2 is not submitted.\033[0m', check3_info)

    if ask('Wanna see opt_hist?'):
        # генерируем таблицу истории оптимизации до проваленного теста (она будет выведена в обратном порядке)
        table_opt_hist = PrettyTable(['surrogate_customer_id', 'relative date', 'channel_code', 'product_code'])
        for item in opt_hist:
            table_opt_hist.add_row(item)

        # print('REVERSED Full optimization history {}'.format('' if check2_flag else '(until failed test)'))
        # print(table_opt_hist)

        # output.sort(key=itemgetter(0))  # восстановление изначального порядка иксов в output
        # print("FROM output[]:")
        # for j in range(len(products)):
        #     i = 0
        #     k = 91
        #     d = 0
        #     index = i * len(
        #         products) * len_customers * period_length + j * len_customers * period_length + k * period_length + d
        #     print(output[index][6], '(index = {})'.format(index))

# TODO: check4: matrix product

print('\n' + 'Step 4: matrix product')
print('\033[31mNot ready yet\033[0m')

# TODO: check5: constraint ratio product

print('\n' + 'Step 5: constraint ratio product')
print('\033[31mNot ready yet\033[0m')

# собираем целевую функцию
objective = 0.0
for item in output:
    objective += -item[5] * item[6]

print("Final objective function: {}".format(objective))


# функция, выводящая только желаемые корни в отформатированном виде
def print_roots(ch=None, prod=None, cust=None, day=None):
    table = PrettyTable(['p (Ordinal)', 'Channel', 'Product', 'Client', 'Day', 'Expectation', 'x'])
    if isinstance(id, list):
        for item in id:
            if isinstance(item, int):
                table.add_row(['?', '?', '?', '?', '?', '?', x[item]])  # TODO: допилить

    if isinstance(cust, int):
        for p in range(len(x)):
            i = p // period_length // len_customers // len(products)
            j = p // period_length // len_customers % len(products)
            k = p // period_length % len_customers  # восстанавливаем k по p в одномерном списке переменных
            d = p % period_length

            if i == ch and k == cust and d == day:  # TODO: Если какой-то из аргументов is None, то не включать его в проверку
                table.add_row([p, i, j, k, d, model[i][j][k], x[p]])

    print(table)

# print_roots(ch=0, cust=91, day=0)  # хочу вывести для 91 кастомера - уже не актуально
