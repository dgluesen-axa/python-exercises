def front_x(words):
  x_list = []
  other_list = []
  for i in range(len(words)):
    if words[i][0] == 'x':
      x_list.append(words[i])
    else:
      other_list.append(words[i])
  x_list.sort()
  other_list.sort()
  result_list = x_list + other_list
  return result_list
