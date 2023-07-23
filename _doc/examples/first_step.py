"""
First steps with pandas_streaming
=================================
 
A few difference between :epkg:`pandas` and *pandas_streaming*.

pandas to pandas_streaming
++++++++++++++++++++++++++
"""

from pandas import DataFrame

df = DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))
df


#############################
# We create a streaming dataframe:


from pandas_streaming.df import StreamingDataFrame

sdf = StreamingDataFrame.read_df(df)
sdf


################################
#

sdf.to_dataframe()


########################################
# Internally, StreamingDataFrame implements an iterator on
# dataframes and then tries to replicate the same interface as
# :class:`pandas.DataFrame` possibly wherever it is possible to
# manipulate data without loading everything into memory.


sdf2 = sdf.concat(sdf)
sdf2.to_dataframe()


###############################
#

m = DataFrame(dict(Y=["a", "b"], Z=[10, 20]))
m


##########################################
#

sdf3 = sdf2.merge(m, left_on="Y", right_on="Y", how="outer")
sdf3.to_dataframe()


############################################
#

sdf2.to_dataframe().merge(m, left_on="Y", right_on="Y", how="outer")


############################################
# The order might be different.


sdftr, sdfte = sdf2.train_test_split(test_size=0.5)
sdfte.head()


############################################
#


sdftr.head()


############################################
# split a big file
# ++++++++++++++++


sdf2.to_csv("example.txt")


############################################
#


new_sdf = StreamingDataFrame.read_csv("example.txt")
new_sdf.train_test_split("example.{}.txt", streaming=False)


############################################
#

import glob

glob.glob("ex*.txt")
