from typing import Optional

def batcher(dataset,
            batch_size: int,
            row_reader=lambda x: (x, None),
            row_continuation= lambda x, c: ({'text':x['text'][c:]}, None),
            epochs:Optional[float]=None,
            allow_short=True,
            fast_forward:int=0):
  """Why build things this way? The 'standard' training that just concatenates strings together never really made sense to me.
  This will take each example and if it is too long, break it and continue it as another sample in the batch.
  If an example would go to a new batch it will be dropped since we don't want to cheat and let a new batch learn from previous batches.

  :param dataset: dataset to batch from
  :param batch_size: guess
  :param row_reader: converts rows from the dataset into a standard format for the model/continuation function.
  :param row_continuation: Function to create a continuation from a sample that is too long.
  :param epochs: I hate epochs but they are easy to think about. Curriculum is a different exp
  :param allow_short: let the final batch be short.
  :param fast_forward: For model re-starts
  :return:
  """
  batch = []
  count = 0
  dataset_len = len(dataset)
  running = True
  while running:

    if count == 0 and fast_forward > 0:
      offset = fast_forward % dataset_len
      count = fast_forward
    else:
      offset = 0
    #Just re-start the dataset
    new_count = 0
    while offset < dataset_len:
      row = dataset[offset]
      continuation = -1
      while continuation is not None:
        if continuation == -1:
          built_row, continuation = row_reader(row)
          count += 1
          offset += 1
          new_count += 1
        else:
          #Gives a chance to 'continue' a row that was too long as another example.
          built_row, continuation = row_continuation(row, continuation)
        batch.append(built_row)
        if epochs is not None and count/len(dataset) >= epochs:
          #Looks like we have see enough examples.
          running = False
          break
        if len(batch) == batch_size:
          yield batch, new_count
          new_count = 0
          #Even if we have more we could get from this sample we want to break continuation for two reasons:
          #1) When a model re-starts this should allow batches to align making stats comparisons a little less fuzzy (not huge but nice)
          #2) We don't want to give the model a chance to 'cheat' by getting an update mid sample.
          # If this is desired behavior then it is a curriculum thing that should be built into the data loading.
          continuation = None
          batch = []


    if len(batch) > 0 and (allow_short or len(batch) == batch_size):
      yield batch, new_count
