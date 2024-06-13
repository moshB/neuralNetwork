import json
from decimal import Decimal, getcontext

import numpy as np
data = {
  "name": "Jason",
  "age": 30,
  "city": [1,2,4]
}

json_string = json.dumps(data)
with open("data.json", "w") as outfile:
  outfile.write(json_string)

# Set the precision to 200 digits
v=[1,2]
u=[1,2,7]
a=np.outer(v,u)
# a=v*u#np.dot(v,u)
print(a)
# Perform calculations with high precision
# num1 = Decimal('1.23456745678543747958709680909980989097853241389')
# num2 = Decimal(3.43564789097865432412433576)
#
# result = num1 * num2*2
#
# # Print the result with 200 digits
# l = [1,8,'4']
# for s, x in l:
#     print(s)
#     print(x)