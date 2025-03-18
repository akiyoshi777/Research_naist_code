import pandas
import tqdm
import time
start_time = time.time()
for i in tqdm.tqdm(range(1000000000)):
    pass
print("--- %s seconds ---" % (time.time() - start_time))