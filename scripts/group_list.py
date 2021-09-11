# Grouping list elements based on frequency
from collections import Counter
  
def group_list(lst):
      
    return list(zip(Counter(lst).keys(), Counter(lst).values()))