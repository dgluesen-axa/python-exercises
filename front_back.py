def front_back(a, b):
  if len(a)%2 == 0:
    a_front = a[:len(a)/2]
    a_back = a[len(a)/2:]
  else:
    a_front = a[:len(a)/2+1]
    a_back = a[len(a)/2+1:]
  if len(b)%2 == 0:
    b_front = b[:len(b)/2]
    b_back = b[len(b)/2:]
  else:
    b_front = b[:len(b)/2+1]
    b_back = b[len(b)/2+1:]
  return a_front + b_front + a_back + b_back
