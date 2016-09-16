def key_func(tupel):
  return tupel[len(tupel)-1]

def sort_last(tuples):
  return sorted(tuples, key=key_func)
