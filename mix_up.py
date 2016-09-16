def mix_up(a, b):
  first_two = a[:2]
  second_two = b[:2]
  return second_two + a[2:] + ' ' + first_two + b[2:]
