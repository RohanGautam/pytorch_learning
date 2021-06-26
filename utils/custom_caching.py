from cassandra.cqltypes import BytesType
from diskcache import FanoutCache, Disk, core
from diskcache.core import io
from io import BytesIO
from diskcache.core import MODE_BINARY
import gzip

# the class we'll make will gzip encode and decode our data while caching
# reading the preprocessed data like this is faster than preprocessing it every time it is run


# class GzipDisk(Disk):
#     def store(self, value, read, key=None):
#         if type(value) is BytesType:
#             if read:
#                 value = value.read()
#                 read = False
#         str_io = BytesIO()
#         # when we write to the file, we can read from the str_io file
#         gz_file = gzip.GzipFile(mode='w', compresslevel=1, fileobj=str_io)

#         for offset in range(0, len(value), 2**30):
#             gz_file.write(value[offset:offset+2**30])
#             gz_file.close()

#             value = str_io.getvalue()

#         return super(GzipDisk, self).store(value, read)

#     def fetch(self, mode, filename, value, read):
#         value = super(GzipDisk, self).fetch(mode, filename, value, read)

#         if mode == MODE_BINARY:
#             str_io = BytesIO(value)
#             gz_file = gzip.GzipFile(mode='r', fileobj=str_io)
#             read_csio = BytesIO()

#             while True:
#                 uncompressed_data = gz_file.read(2**30)
#                 if uncompressed_data:
#                     read_csio.write(uncompressed_data)
#                 else:
#                     break

#             value = read_csio.getvalue()

#         return value


def getCache(scope_str):
    return FanoutCache('data_unversioned/cache/'+scope_str,
                       disk=Disk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11
                       )
