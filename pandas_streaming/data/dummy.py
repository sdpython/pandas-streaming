from pandas import DataFrame
from ..df import StreamingDataFrame


def dummy_streaming_dataframe(n, chunksize=10, asfloat=False, **cols):
    """
    Returns a dummy streaming dataframe
    mostly for unit test purposes.

    :param n: number of rows
    :param chunksize: chunk size
    :param asfloat: use random float and not random int
    :param cols: additional columns
    :return: a @see cl StreamingDataFrame
    """
    if asfloat:
        df = DataFrame(
            dict(
                cfloat=[_ + 0.1 for _ in range(0, n)],
                cstr=[f"s{i}" for i in range(0, n)],
            )
        )
    else:
        df = DataFrame(
            dict(cint=list(range(0, n)), cstr=[f"s{i}" for i in range(0, n)])
        )
    for k, v in cols.items():
        df[k] = v
    return StreamingDataFrame.read_df(df, chunksize=chunksize)
