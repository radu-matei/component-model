import definitions
from definitions import *

def equal_modulo_string_encoding(s, t):
  if isinstance(s, (bool,int,float,str)) and isinstance(t, (bool,int,float,str)):
    return s == t
  if isinstance(s, tuple) and isinstance(t, tuple):
    if s == () and t == ():
      return True
    assert(isinstance(s[0], str))
    assert(isinstance(t[0], str))
    return s[0] == t[0]
  if isinstance(s, dict) and isinstance(t, dict):
    return all(equal_modulo_string_encoding(sv,tv) for sv,tv in zip(s.values(), t.values(), strict=True))
  if isinstance(s, list) and isinstance(t, list):
    return all(equal_modulo_string_encoding(sv,tv) for sv,tv in zip(s, t, strict=True))
  assert(False)

class Heap:
  def __init__(self, arg):
    self.memory = bytearray(arg)
    self.last_alloc = 0

  def realloc(self, original_ptr, original_size, alignment, new_size):
    if original_ptr != 0 and new_size < original_size:
      return align_to(original_ptr, alignment)
    ret = align_to(self.last_alloc, alignment)
    self.last_alloc = ret + new_size
    if self.last_alloc > len(self.memory):
      print('oom: have {} need {}'.format(len(self.memory), self.last_alloc))
      trap()
    self.memory[ret : ret + original_size] = self.memory[original_ptr : original_ptr + original_size]
    return ret

def mk_opts(memory, encoding, realloc, post_return):
  opts = Opts()
  opts.memory = memory
  opts.string_encoding = encoding
  opts.realloc = realloc
  opts.post_return = post_return
  return opts

def mk_str(s):
  return (s, 'utf8', len(s.encode('utf-8')))

def mk_tup(*a):
  def mk_tup_rec(x):
    if isinstance(x, list):
      return { str(i):mk_tup_rec(v) for i,v in enumerate(x) }
    return x
  return { str(i):mk_tup_rec(v) for i,v in enumerate(a) }

def fail(msg):
  raise BaseException(msg)

def test(t, vals_to_lift, v,
         opts = mk_opts(bytearray(), 'utf8', None, None),
         dst_encoding = None,
         lower_t = None,
         lower_v = None):
  def test_name():
    return "test({},{},{}):".format(t, vals_to_lift, v)

  vi = ValueIter([Value(ft, v) for ft,v in zip(flatten(t), vals_to_lift, strict=True)])

  if v is None:
    try:
      got = lift(opts, vi, t)
      fail("{} expected trap, but got {}".format(test_name(), got))
    except Trap:
      return

  got = lift(opts, vi, t)
  assert(vi.i == len(vi.values))
  if got != v:
    fail("{} initial lift() expected {} but got {}".format(test_name(), v, got))

  if lower_t is None:
    lower_t = t
  if lower_v is None:
    lower_v = v

  heap = Heap(5*len(opts.memory))
  if dst_encoding is None:
    dst_encoding = opts.string_encoding
  opts = mk_opts(heap.memory, dst_encoding, heap.realloc, None)
  lowered_vals = lower(opts, v, lower_t)
  assert(flatten(lower_t) == list(map(lambda v: v.t, lowered_vals)))

  vi = ValueIter(lowered_vals)
  got = lift(opts, vi, lower_t)
  if not equal_modulo_string_encoding(got, lower_v):
    fail("{} re-lift expected {} but got {}".format(test_name(), lower_v, got))

test(Unit(), [], {})
test(Record([Field('x',U8()), Field('y',U16()), Field('z',U32())]), [1,2,3], {'x':1,'y':2,'z':3})
test(Tuple([Tuple([U8(),U8()]),U8()]), [1,2,3], {'0':{'0':1,'1':2},'1':3})
t = Flags(['a','b'])
test(t, [0], {'a':False,'b':False})
test(t, [2], {'a':False,'b':True})
test(t, [3], {'a':True,'b':True})
test(t, [4], None)
test(Flags([str(i) for i in range(33)]), [0xffffffff,0x1], { str(i):True for i in range(33) })
t = Variant([Case('x',U8()),Case('y',Float32()),Case('z',Unit())])
test(t, [0,42], {'x': 42})
test(t, [0,256], None)
test(t, [1,0x4048f5c3], {'y': 3.140000104904175})
test(t, [2,0xffffffff], {'z': {}})
t = Union([U32(),U64()])
test(t, [0,42], {'0':42})
test(t, [0,(1<<35)], None)
test(t, [1,(1<<35)], {'1':(1<<35)})
t = Union([Float32(), U64()])
test(t, [0,0x4048f5c3], {'0': 3.140000104904175})
test(t, [0,(1<<35)], None)
test(t, [1,(1<<35)], {'1': (1<<35)})
t = Union([Float64(), U64()])
test(t, [0,0x40091EB851EB851F], {'0': 3.14})
test(t, [0,(1<<35)], {'0': 1.69759663277e-313})
test(t, [1,(1<<35)], {'1': (1<<35)})
t = Union([U8()])
test(t, [0,42], {'0':42})
test(t, [1,256], None)
test(t, [0,256], None)
t = Union([Tuple([U8(),Float32()]), U64()])
test(t, [0,42,3.14], {'0': {'0':42, '1':3.14}})
test(t, [1,(1<<35),0], {'1': (1<<35)})
t = Option(Float32())
test(t, [0,3.14], {'none':{}})
test(t, [1,3.14], {'some':3.14})
t = Expected(U8(),U32())
test(t, [0, 42], {'ok':42})
test(t, [1, 1000], {'error':1000})
t = Variant([Case('w',U8()), Case('x',U8(),'w'), Case('y',U8()), Case('z',U8(),'x')])
test(t, [0, 42], {'w':42})
test(t, [1, 42], {'x|w':42})
test(t, [2, 42], {'y':42})
test(t, [3, 42], {'z|x|w':42})
t2 = Variant([Case('w',U8())])
test(t, [0, 42], {'w':42}, lower_t=t2, lower_v={'w':42})
test(t, [1, 42], {'x|w':42}, lower_t=t2, lower_v={'w':42})
test(t, [3, 42], {'z|x|w':42}, lower_t=t2, lower_v={'w':42})

def test_pairs(t, pairs):
  for arg,expect in pairs:
    test(t, [arg], expect)

test_pairs(Bool(), [(0,False),(1,True),(2,True),(4294967295,True)])
test_pairs(U8(), [(127,127),(128,128),(255,255),(256,None),
                  (4294967295,None),(4294967168,None),(4294967167,None)])
test_pairs(S8(), [(127,127),(128,None),(255,None),(256,None),
                  (4294967295,-1),(4294967168,-128),(4294967167,None)])
test_pairs(U16(), [(32767,32767),(32768,32768),(65535,65535),(65536,None),
                   ((1<<32)-1,None),((1<<32)-32768,None),((1<<32)-32769,None)])
test_pairs(S16(), [(32767,32767),(32768,None),(65535,None),(65536,None),
                   ((1<<32)-1,-1),((1<<32)-32768,-32768),((1<<32)-32769,None)])
test_pairs(U32(), [((1<<31)-1,(1<<31)-1),(1<<31,1<<31),(((1<<32)-1),(1<<32)-1)])
test_pairs(S32(), [((1<<31)-1,(1<<31)-1),(1<<31,-(1<<31)),((1<<32)-1,-1)])
test_pairs(U64(), [((1<<63)-1,(1<<63)-1), (1<<63,1<<63), ((1<<64)-1,(1<<64)-1)])
test_pairs(S64(), [((1<<63)-1,(1<<63)-1), (1<<63,-(1<<63)), ((1<<64)-1,-1)])
test_pairs(Float32(), [(3.14,3.14)])
test_pairs(Float64(), [(3.14,3.14)])
test_pairs(Char(), [(0,'\x00'), (65,'A'), (0xD7FF,'\uD7FF'), (0xD800,None), (0xDFFF,None)])
test_pairs(Char(), [(0xE000,'\uE000'), (0x10FFFF,'\U0010FFFF'), (0x110000,None), (0xFFFFFFFF,None)])
test_pairs(Enum(['a','b']), [(0,{'a':{}}), (1,{'b':{}}), (2,None)])

def test_string_internal(src_encoding, dst_encoding, s, encoded, encoding, utf16_flag = False):
  heap = Heap(len(encoded))
  heap.memory[:] = encoded[:]
  opts = mk_opts(heap.memory, src_encoding, None, None)
  v = (s, encoding, len(encoded))
  if utf16_flag:
    arg2 = (len(encoded) >> 1) | (1 << 31)
  else:
    arg2 = len(encoded)
  test(String(), [0, arg2], v, opts, dst_encoding)

def test_string(src_encoding, dst_encoding, s):
  if src_encoding == 'latin1+utf16':
    try:
      encoded = s.encode('latin-1')
      test_string_internal(src_encoding, dst_encoding, s, encoded, 'latin1')
    except UnicodeEncodeError:
      pass
    encoded = s.encode('utf-16-le')
    test_string_internal(src_encoding, dst_encoding, s, encoded, 'utf16', utf16_flag = True)
  else:
    encoded = s.encode(to_python_encoding(src_encoding))
    test_string_internal(src_encoding, dst_encoding, s, encoded, src_encoding)

encodings = ['utf8', 'utf16', 'latin1+utf16']

fun_strings = ['', 'a', 'hi', '\x00', 'a\x00b', '\x80', '\x80b', 'ab\xefc',
               '\u01ffy', 'xy\u01ff', 'a\ud7ffb', 'a\u02ff\u03ff\u04ffbc',
               '\uf123', '\uf123\uf123abc', 'abcdef\uf123']

for src_encoding in encodings:
  for dst_encoding in encodings:
    for s in fun_strings:
      test_string(src_encoding, dst_encoding, s)

def test_heap(t, expect, args, byte_array):
  heap = Heap(byte_array)
  opts = mk_opts(heap.memory, 'utf8', None, None)
  test(t, args, expect, opts)

test_heap(List(Unit()), [{},{},{}], [0,3], [])
test_heap(List(Bool()), [True,False,True], [0,3], [1,0,1])
test_heap(List(Bool()), [True,False,True], [3,3], [0xff,0xff,0xff, 1,0,1])
test_heap(List(U8()), [1,2,3], [0,3], [1,2,3])
test_heap(List(U16()), [1,2,3], [0,3], [1,0, 2,0, 3,0 ])
test_heap(List(U32()), [1,2,3], [0,3], [1,0,0,0, 2,0,0,0, 3,0,0,0])
test_heap(List(U64()), [1,2], [0,2], [1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0])
test_heap(List(S8()), [-1,-2,-3], [0,3], [0xff,0xfe,0xfd])
test_heap(List(S16()), [-1,-2,-3], [0,3], [0xff,0xff, 0xfe,0xff, 0xfd,0xff])
test_heap(List(S32()), [-1,-2,-3], [0,3], [0xff,0xff,0xff,0xff, 0xfe,0xff,0xff,0xff, 0xfd,0xff,0xff,0xff])
test_heap(List(S64()), [-1,-2], [0,2], [0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff, 0xfe,0xff,0xff,0xff,0xff,0xff,0xff,0xff])
test_heap(List(Char()), ['A','B','c'], [0,3], [65,00,00,00, 66,00,00,00, 99,00,00,00])
test_heap(List(String()), [mk_str("hi"),mk_str("wat")], [0,2],
          [16,0,0,0, 2,0,0,0, 21,0,0,0, 3,0,0,0,
           ord('h'), ord('i'),   0xf,0xf,0xf,   ord('w'), ord('a'), ord('t')])
test_heap(List(List(U8())), [[3,4,5],[],[6,7]], [0,3],
          [24,0,0,0, 3,0,0,0, 0,0,0,0, 0,0,0,0, 27,0,0,0, 2,0,0,0,
          3,4,5,  6,7])
test_heap(List(Tuple([U8(),U8(),U16(),U32()])), [mk_tup(6,7,8,9),mk_tup(4,5,6,7)], [0,2],
          [6, 7, 8,0, 9,0,0,0,   4, 5, 6,0, 7,0,0,0])
test_heap(List(Tuple([U8(),U16(),U8(),U32()])), [mk_tup(6,7,8,9),mk_tup(4,5,6,7)], [0,2],
          [6,0xff, 7,0, 8,0xff,0xff,0xff, 9,0,0,0,   4,0xff, 5,0, 6,0xff,0xff,0xff, 7,0,0,0])
test_heap(List(Tuple([U16(),U8()])), [mk_tup(6,7),mk_tup(8,9)], [0,2],
          [6,0, 7, 0x0ff, 8,0, 9, 0xff])
test_heap(List(Tuple([Tuple([U16(),U8()]),U8()])), [mk_tup([4,5],6),mk_tup([7,8],9)], [0,2],
          [4,0, 5, 6,  7,0, 8, 9])
test_heap(List(Union([Unit(),U8(),Tuple([U8(),U16()])])), [{'0':{}}, {'1':42}, {'2':mk_tup(6,7)}], [0,3],
          [0,0xff,0xff,0xff,0xff,0xff,  1,0xff,42,0xff,0xff,0xff,  2,0xff,6,0xff,7,0])
test_heap(List(Union([U32(),U8()])), [{'0':256}, {'1':42}], [0,2],
          [0,0xff,0xff,0xff,0,1,0,0,  1,0xff,0xff,0xff,42,0xff,0xff,0xff])
test_heap(List(Tuple([Union([U8(),Tuple([U16(),U8()])]),U8()])),
          [mk_tup({'1':mk_tup(5,6)},7),mk_tup({'0':8},9)], [0,2],
          [1,0xff,5,0,6,7,  0,0xff,8,0xff,0xff,9])
test_heap(List(Union([U8()])), [{'0':6},{'0':7},{'0':8}], [0,3],
          [0,6, 0,7, 0,8])
t = List(Flags(['a','b']))
test_heap(t, [{'a':False,'b':False},{'a':False,'b':True},{'a':True,'b':True}], [0,3],
          [0,2,3])
test_heap(t, None, [0,3],
          [0,2,4])
t = List(Flags([str(i) for i in range(9)]))
test_heap(t, [{ str(i):b for i in range(9) } for b in [True,False]], [0,2],
          [0xff,0x1, 0,0])
test_heap(t, None, [0,2],
          [0xff,0x3, 0,0])
t = List(Flags([str(i) for i in range(17)]))
test_heap(t, [{ str(i):b for i in range(17) } for b in [True,False]], [0,2],
          [0xff,0xff,0x1,0, 0,0,0,0])
test_heap(t, None, [0,2],
          [0xff,0xff,0x3,0, 0,0,0,0])
t = List(Flags([str(i) for i in range(33)]))
test_heap(t, [{ str(i):b for i in range(33) } for b in [True,False]], [0,2],
          [0xff,0xff,0xff,0xff,0x1,0,0,0, 0,0,0,0,0,0,0,0])
test_heap(t, None, [0,2],
          [0xff,0xff,0xff,0xff,0x3,0,0,0, 0,0,0,0,0,0,0,0])

def test_roundtrip(t, v):
  before = definitions.MAX_RESULTS
  definitions.MAX_RESULTS = 16

  ft = Func([t],t)
  callee = FuncInst()
  callee.instance = Instance()
  callee.func = lambda x: x

  callee_heap = Heap(1000)
  callee_opts = mk_opts(callee_heap.memory, 'utf8', callee_heap.realloc, lambda: ())
  lifted_callee = FuncInst()
  lifted_callee.instance = callee.instance
  lifted_callee.func = lambda args: call_in(callee_opts, callee, ft, args)

  caller_heap = Heap(1000)
  caller_instance = Instance()
  caller_opts = mk_opts(caller_heap.memory, 'utf8', caller_heap.realloc, None)

  flat_args = lower(caller_opts, v, t)
  flat_results = call_out(caller_opts, caller_instance, lifted_callee, ft, flat_args)
  got = lift(caller_opts, ValueIter(flat_results), t)

  if got != v:
    fail("test_roundtrip({},{},{}) got {}".format(t, v, caller_args, got))

  assert(caller_instance.may_leave and caller_instance.may_enter)
  assert(callee.instance.may_leave and callee.instance.may_enter)
  definitions.MAX_RESULTS = before

test_roundtrip(S8(), -1)
test_roundtrip(Tuple([U16(),U16()]), mk_tup(3,4))
test_roundtrip(List(String()), [mk_str("hello there")])
test_roundtrip(List(List(String())), [[mk_str("one"),mk_str("two")],[mk_str("three")]])
test_roundtrip(List(Option(Tuple([String(),U16()]))), [{'some':mk_tup(mk_str("answer"),42)}])

print("All tests passed")
