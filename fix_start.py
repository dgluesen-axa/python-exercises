def fix_start(s):
  first = s[0]
  return first + s[1:].replace(first, '*')
