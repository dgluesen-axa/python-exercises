def verbing(s):
  if len(s) > 2:
    if s[len(s)-3:len(s)] == 'ing':
      result = s + 'ly'
    else:
      result = s + 'ing'
  else:
    result = s
  return result
