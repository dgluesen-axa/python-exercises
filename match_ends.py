def match_ends(words):
  count = 0
  for i in range(len(words)):
    if len(words[i]) > 1:
      if words[i][0] == words[i][len(words[i])-1]:
        count += 1
  return count
