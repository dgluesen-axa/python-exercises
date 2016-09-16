def both_ends(s):
  if len(s) < 2:
    result = ''
  else:
    result = s[0] + s[1] + s[len(s)-2] + s[len(s)-1]
  return result
