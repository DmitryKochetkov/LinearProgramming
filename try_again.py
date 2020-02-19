import csv
from cvxopt.modeling import variable, op, matrix
from cvxopt.glpk import ilp
from numpy import array
from operator import itemgetter

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

output = []

# Data types conversion
for score in scores:
    score[0] = int(score[0])
    score[1] = int(score[1])
    score[2] = float(score[2])
    output.append([score[0], score[1], 0])

for model in dict_model:
    model[0] = int(model[0])

for product in products:
    product[0] = int(product[0])
    product[2] = float(product[2])

products.sort(key=itemgetter(0))

for i in range(0, len(channels)):
    channels[i] = channels[i][0]

print("Dict_Model ({} items):".format(len(dict_model)), dict_model[:3], "...")
print("Products ({} items):".format(len(products)), products[:3], "...")
print("Channels ({} items):".format(len(channels)), channels[:3], "...")


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


# Preparing data

print("Scores ({} items):".format(len(scores)), scores[:3], "...")

# 2. Creating model
model = []

for channel in channels:
    model.append(list())
    for product in products:
        model[channels.index(channel)].append(list())
        # And here I tried dict()
        model[channels.index(channel)][products.index(product)] = dict()

# 3. Filling model
for score in scores:
    this_client = score[0]
    this_model = score[1]
    this_probability = score[2]

    this_channel = channel_by_model(this_model)
    this_product = product_by_model(this_model)

    model[this_channel][this_product][this_client] = this_probability * products[this_product][2]

print('\nMODEL')
# maybe better tree?
for i in range(0, len(model)):
    print('{}:'.format(channels[i]))
    for j in range(0, len(model[i])):
        print('\t{}:'.format(products[j]))
        for key, value in dict(model[i][j]).items():
            print('\t\tcustomer {}: {}'.format(key, value))

# solution

x = []  # variables

for i in range(0, len(model)):
    x.append(list())
    for j in range(0, len(model[i])):
        x[i].append(list())
        size = len(dict(model[i][j]).keys())  # size of new variable vector
        x[i][j] = variable(151, 'x{}{}'.format(i, j))

f = 0

for i in range(0, len(model)):
    for j in range(0, len(model[i])):
        for k in dict(model[i][j]).keys():
            f += model[i][j][k] * x[i][j][k]

# output

print('\nOUTPUT')
print('channel | product | client | x')

# for out in output:
#     out[2] = required x
#     print("%6d | %5d | %d" % (out[0], out[1], out[2]))