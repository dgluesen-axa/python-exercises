def not_bad(s):
  part1 = 'not'
  part2 = 'bad'
  if s.find(part1) < s.find(part2):
    result = s[:s.find(part1)] + 'good' + s[s.find(part2)+len(part2):]
  else:
    result = s
  return result
