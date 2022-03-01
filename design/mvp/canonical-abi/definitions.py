# Boilerplate

import math
import struct
import types
from dataclasses import dataclass

class Trap(BaseException):
  pass

def trap():
  raise Trap()

def trap_if(cond):
  if cond:
    raise Trap()

def assert_unreachable(v):
  print("Unreachable ({})".format(v))
  assert(False)

def to_python_encoding(encoding):
  match encoding:
    case 'utf8'   : return 'utf-8'
    case 'utf16'  : return 'utf-16-le'
    case 'latin1' : return 'latin-1'
    case _        : assert_unreachable(encoding)

class Unit: pass
class Bool: pass
class S8: pass
class U8: pass
class S16: pass
class U16: pass
class S32: pass
class U32: pass
class S64: pass
class U64: pass
class Float32: pass
class Float64: pass
class Char: pass
class String: pass

@dataclass
class List:
  t: any

@dataclass
class Field:
  label: str
  t: any

@dataclass
class Record:
  fields: [Field]

@dataclass
class Tuple:
  ts: [any]

@dataclass
class Flags:
  labels: [str]

@dataclass
class Case:
  label: str
  t: any
  defaults_to: str = None

@dataclass
class Variant:
  cases: [Case]

@dataclass
class Enum:
  labels: [str]

@dataclass
class Union:
  ts: [any]

@dataclass
class Option:
  t: any

@dataclass
class Expected:
  ok: any
  error: any

@dataclass
class Func:
  params: [any]
  result: any

# Despecialization

def despecialize(t):
  match t:
    case Tuple(ts)           : return Record([ Field(str(i), t) for i,t in enumerate(ts) ])
    case Unit()              : return Record([])
    case Union(ts)           : return Variant([ Case(str(i), t) for i,t in enumerate(ts) ])
    case Enum(labels)        : return Variant([ Case(l, Unit()) for l in labels ])
    case Option(t)           : return Variant([ Case("none", Unit()), Case("some", t) ])
    case Expected(ok, error) : return Variant([ Case("ok", ok), Case("error", error) ])
    case _                   : return t

# Alignment

def alignment(t):
  match despecialize(t):
    case Bool()             : return 1
    case S8() | U8()        : return 1
    case S16() | U16()      : return 2
    case S32() | U32()      : return 4
    case S64() | U64()      : return 8
    case Float32()          : return 4
    case Float64()          : return 8
    case Char()             : return 4
    case String() | List(_) : return 4
    case Record(fields)     : return max_alignment(types_of(fields))
    case Variant(cases)     : return max_alignment(types_of(cases) + [discriminant_type(cases)])
    case Flags(labels)      : return alignment_flags(labels)
    case _                  : assert_unreachable(t)

def max_alignment(ts):
  a = 1
  for t in ts:
    a = max(a, alignment(t))
  return a

def types_of(fields_or_cases):
  return [x.t for x in fields_or_cases]

#

def discriminant_type(cases):
  n = len(cases)
  assert(0 < n < (1 << 32))
  match math.ceil(math.log2(n)/8):
    case 0: return U8()
    case 1: return U8()
    case 2: return U16()
    case 3: return U32()
    case _: return assert_unreachable(n)

#

def alignment_flags(labels):
  n = len(labels)
  if n <= 8: return 1
  if n <= 16: return 2
  return 4

# Size

def elem_size(t):
  return align_to(byte_size(t), alignment(t))

def align_to(ptr, alignment):
  return math.ceil(ptr / alignment) * alignment

def byte_size(t):
  match despecialize(t):
    case Bool()             : return 1
    case S8() | U8()        : return 1
    case S16() | U16()      : return 2
    case S32() | U32()      : return 4
    case S64() | U64()      : return 8
    case Float32()          : return 4
    case Float64()          : return 8
    case Char()             : return 4
    case String() | List(_) : return 8
    case Record(fields)     : return byte_size_record(fields)
    case Variant(cases)     : return byte_size_variant(cases)
    case Flags(labels)      : return byte_size_flags(labels)
    case _                  : assert_unreachable(t)

def byte_size_record(fields):
  s = 0
  for f in fields:
    s = align_to(s, alignment(f.t))
    s += byte_size(f.t)
  return s

def byte_size_variant(cases):
  s = byte_size(discriminant_type(cases))
  s = align_to(s, max_alignment(types_of(cases)))
  cs = 0
  for c in cases:
    cs = max(cs, byte_size(c.t))
  return s + cs

def byte_size_flags(labels):
  n = len(labels)
  if n <= 8: return 1
  if n <= 16: return 2
  return 4 * math.ceil(n / 32)

# Loading

class Opts:
  memory: bytearray
  string_encoding: str
  realloc: types.FunctionType
  post_return: types.FunctionType

def load(opts, ptr, t):
  assert(ptr == align_to(ptr, alignment(t)))
  match despecialize(t):
    case Bool()         : return bool(load_int(opts, ptr, 1))
    case U8()           : return load_int(opts, ptr, 1)
    case U16()          : return load_int(opts, ptr, 2)
    case U32()          : return load_int(opts, ptr, 4)
    case U64()          : return load_int(opts, ptr, 8)
    case S8()           : return load_int(opts, ptr, 1, signed=True)
    case S16()          : return load_int(opts, ptr, 2, signed=True)
    case S32()          : return load_int(opts, ptr, 4, signed=True)
    case S64()          : return load_int(opts, ptr, 8, signed=True)
    case Float32()      : return canonicalize(reinterpret_i32_as_float(load_int(opts, ptr, 4)))
    case Float64()      : return canonicalize(reinterpret_i64_as_float(load_int(opts, ptr, 8)))
    case Char()         : return i32_to_char(opts, load_int(opts, ptr, 4))
    case String()       : return load_string(opts, ptr)
    case List(t)        : return load_list(opts, ptr, t)
    case Record(fields) : return load_record(opts, ptr, fields)
    case Variant(cases) : return load_variant(opts, ptr, cases)
    case Flags(labels)  : return load_flags(opts, ptr, labels)
    case _              : assert_unreachable(t)

#

def load_int(opts, ptr, nbytes, signed = False):
  trap_if(ptr + nbytes > len(opts.memory))
  return int.from_bytes(opts.memory[ptr : ptr + nbytes], 'little', signed=signed)

#

def reinterpret_i32_as_float(i):
  return struct.unpack('!f', struct.pack('!I', i))[0]

def reinterpret_i64_as_float(i):
  return struct.unpack('!d', struct.pack('!Q', i))[0]

def canonicalize(f):
  if math.isnan(f):
    return reinterpret_i64_as_float(0x7ff8000000000000)
  return f

#

def i32_to_char(opts, i):
  trap_if(i >= 0x110000)
  trap_if(0xD800 <= i <= 0xDFFF)
  return chr(i)

def load_string(opts, ptr):
  begin = load_int(opts, ptr, 4)
  byte_length = load_int(opts, ptr + 4, 4)
  return load_string_from_range(opts, begin, byte_length)

def load_string_from_range(opts, ptr, byte_length):
  match opts.string_encoding:
    case 'latin1+utf16':
      if bool(byte_length & (1 << 31)):
        byte_length = (byte_length & ~(1 << 31)) << 1
        encoding = 'utf16'
      else:
        encoding = 'latin1'
    case 'utf8' | 'utf16':
      encoding = opts.string_encoding
    case _:
      assert_unreachable(opts.string_encoding)
  trap_if(ptr + byte_length > len(opts.memory))
  try:
    s = opts.memory[ptr : ptr + byte_length].decode(to_python_encoding(encoding))
  except UnicodeError:
    trap()
  return (s, encoding, byte_length)

def load_list(opts, ptr, elem_type):
  begin = load_int(opts, ptr, 4)
  length = load_int(opts, ptr + 4, 4)
  return load_list_from_range(opts, begin, length, elem_type)

def load_list_from_range(opts, ptr, length, elem_type):
  trap_if(ptr + length * elem_size(elem_type) > len(opts.memory))
  a = []
  for i in range(length):
    a.append(load(opts, ptr + i * elem_size(elem_type), elem_type))
  return a

def load_record(opts, ptr, fields):
  record = {}
  for field in fields:
    ptr = align_to(ptr, alignment(field.t))
    record[field.label] = load(opts, ptr, field.t)
    ptr += byte_size(field.t)
  return record

def load_variant(opts, ptr, cases):
  disc_size = byte_size(discriminant_type(cases))
  disc = load_int(opts, ptr, disc_size)
  ptr += disc_size
  trap_if(disc >= len(cases))
  case = cases[disc]
  ptr = align_to(ptr, max_alignment(types_of(cases)))
  return { case_label_with_defaults(case, cases): load(opts, ptr, case.t) }

def case_label_with_defaults(case, cases):
  label = case.label
  assert(label.find('|') == -1)
  while case.defaults_to is not None:
    case = cases[find_case(case.defaults_to, cases)]
    label += '|' + case.label
  return label

def find_case(label, cases):
  matches = [i for i,c in enumerate(cases) if c.label == label]
  assert(len(matches) <= 1)
  if len(matches) == 1:
    return matches[0]
  return -1

def load_flags(opts, ptr, labels):
  i = load_int(opts, ptr, byte_size_flags(labels))
  return load_flags_from_bigint(i, labels)

def load_flags_from_bigint(i, labels):
  record = {}
  for l in labels:
    record[l] = bool(i & 1)
    i >>= 1
  trap_if(i)
  return record

# Storing

def store(opts, v, t, ptr):
  assert(ptr == align_to(ptr, alignment(t)))
  match despecialize(t):
    case Bool()         : store_int(opts, int(v), ptr, 1)
    case U8()           : store_int(opts, v, ptr, 1)
    case U16()          : store_int(opts, v, ptr, 2)
    case U32()          : store_int(opts, v, ptr, 4)
    case U64()          : store_int(opts, v, ptr, 8)
    case S8()           : store_int(opts, v, ptr, 1, signed=True)
    case S16()          : store_int(opts, v, ptr, 2, signed=True)
    case S32()          : store_int(opts, v, ptr, 4, signed=True)
    case S64()          : store_int(opts, v, ptr, 8, signed=True)
    case Float32()      : store_int(opts, reinterpret_float_as_i32(v), ptr, 4)
    case Float64()      : store_int(opts, reinterpret_float_as_i64(v), ptr, 8)
    case Char()         : store_int(opts, char_to_int(v), ptr, 4)
    case String()       : store_string(opts, v, ptr)
    case List(t)        : store_list(opts, v, ptr, t)
    case Record(fields) : store_record(opts, v, ptr, fields)
    case Variant(cases) : store_variant(opts, v, ptr, cases)
    case Flags(labels)  : store_flags(opts, v, ptr, labels)
    case _              : assert_unreachable(t)

def store_int(opts, v, ptr, nbytes, signed = False):
  trap_if(ptr + nbytes > len(opts.memory))
  opts.memory[ptr : ptr + nbytes] = int.to_bytes(v, nbytes, 'little', signed=signed)

def reinterpret_float_as_i32(f):
  f = canonicalize(f)
  return struct.unpack('!I', struct.pack('!f', f))[0]

def reinterpret_float_as_i64(f):
  f = canonicalize(f)
  return struct.unpack('!Q', struct.pack('!d', f))[0]

def char_to_int(c):
  i = ord(c)
  assert(0 <= i <= 0xD7FF or 0xD800 <= i <= 0x10FFFF)
  return i

def store_string(opts, v, ptr):
  begin, byte_length = store_string_into_range(opts, v)
  store_int(opts, begin, ptr, 4)
  store_int(opts, byte_length, ptr + 4, 4)

def store_string_into_range(opts, v):
  src, src_encoding, src_byte_length = v
  match opts.string_encoding:
    case 'utf8':
      match src_encoding:
        case 'utf8'   : return store_string_copy(opts, src, src_byte_length, 'utf8')
        case 'utf16'  : return store_utf16_to_utf8(opts, src, src_byte_length)
        case 'latin1' : return store_latin1_to_utf8(opts, src, src_byte_length)
    case 'utf16':
      match src_encoding:
        case 'utf8'   : return store_utf8_to_utf16(opts, src, src_byte_length)
        case 'utf16'  : return store_string_copy(opts, src, src_byte_length, 'utf16')
        case 'latin1' : return store_string_copy(opts, src, src_byte_length, 'utf16', inflation = 2)
    case 'latin1+utf16':
      match src_encoding:
        case 'utf8'   : return store_utf8_to_latin1_or_utf16(opts, src, src_byte_length)
        case 'utf16'  : return store_utf16_to_latin1_or_utf16(opts, src, src_byte_length)
        case 'latin1' : return store_string_copy(opts, src, src_byte_length, 'latin1')
    case _            : assert_unreachable(opts.string_encoding)

def store_string_copy(opts, src, src_byte_length, dst_encoding, inflation = 1):
  ptr = opts.realloc(0, 0, 1, src_byte_length * inflation)
  encoded = src.encode(to_python_encoding(dst_encoding))
  opts.memory[ptr : ptr + len(encoded)] = encoded
  return (ptr, len(encoded))

def store_utf16_to_utf8(opts, src, src_byte_length):
  optimistic_size = src_byte_length >> 1
  worst_case_size = optimistic_size * 3
  return store_string_transcode(opts, src, 'utf8', optimistic_size, worst_case_size)

def store_latin1_to_utf8(opts, src, src_byte_length):
  optimistic_size = src_byte_length
  worst_case_size = optimistic_size * 2
  return store_string_transcode(opts, src, 'utf8', optimistic_size, worst_case_size)

def store_utf8_to_utf16(opts, src, src_byte_length):
  optimistic_size = src_byte_length * 2
  worst_case_size = optimistic_size
  return store_string_transcode(opts, src, 'utf16', optimistic_size, worst_case_size)

def store_string_transcode(opts, src, dst_encoding, optimistic_size, worst_case_size):
  ptr = opts.realloc(0, 0, 1, optimistic_size)
  encoded = src.encode(to_python_encoding(dst_encoding))
  bytes_copied = min(len(encoded), optimistic_size)
  opts.memory[ptr : ptr + bytes_copied] = encoded[0 : bytes_copied]
  if bytes_copied < optimistic_size:
    ptr = opts.realloc(ptr, optimistic_size, 1, bytes_copied)
  elif bytes_copied < len(encoded):
    ptr = opts.realloc(ptr, optimistic_size, 1, worst_case_size)
    opts.memory[ptr+bytes_copied : ptr+len(encoded)] = encoded[bytes_copied:]
    if worst_case_size > len(encoded):
      ptr = opts.realloc(ptr, worst_case_size, 1, len(encoded))
  return (ptr, len(encoded))

def store_utf8_to_latin1_or_utf16(opts, src, src_byte_length):
  ptr = opts.realloc(0, 0, 1, src_byte_length)
  dst_byte_length = 0
  for usv in src:
    if ord(usv) < (1 << 8):
      opts.memory[ptr + dst_byte_length] = ord(usv)
      dst_byte_length += 1
    else:
      worst_case_size = 2 * src_byte_length
      ptr = opts.realloc(ptr, src_byte_length, 1, worst_case_size)
      for j in range(dst_byte_length-1, -1, -1):
        opts.memory[ptr + 2*j] = opts.memory[ptr + j]
        opts.memory[ptr + 2*j + 1] = 0
      encoded = src.encode(to_python_encoding('utf16'))
      opts.memory[ptr+2*dst_byte_length : ptr+len(encoded)] = encoded[2*dst_byte_length:]
      if worst_case_size != len(encoded):
        ptr = opts.realloc(ptr, worst_case_size, 1, len(encoded))
      return (ptr, (len(encoded) >> 1) | (1 << 31))
  if dst_byte_length < src_byte_length:
    ptr = opts.realloc(ptr, src_byte_length, 1, dst_byte_length)
  return (ptr, dst_byte_length)

def store_utf16_to_latin1_or_utf16(opts, src, src_byte_length):
  ptr = opts.realloc(0, 0, 1, src_byte_length)
  encoded = src.encode(to_python_encoding('utf16'))
  opts.memory[ptr : ptr+len(encoded)] = encoded
  if any(ord(c) >= (1 << 8) for c in src):
    return (ptr, (len(encoded) >> 1) | (1 << 31))
  latin1_size = len(encoded) >> 1
  for i in range(latin1_size):
    opts.memory[ptr + i] = opts.memory[ptr + 2*i]
  ptr = opts.realloc(ptr, src_byte_length, 1, latin1_size)
  return (ptr, latin1_size)

def store_list(opts, v, ptr, elem_type):
  begin, length = store_list_into_range(opts, v, elem_type)
  store_int(opts, begin, ptr, 4)
  store_int(opts, length, ptr + 4, 4)

def store_list_into_range(opts, v, elem_type):
  byte_length = len(v) * elem_size(elem_type)
  ptr = opts.realloc(0, 0, alignment(elem_type), byte_length)
  trap_if(ptr + byte_length > len(opts.memory))
  for (i, e) in enumerate(v):
    store(opts, e, elem_type, ptr + i * elem_size(elem_type))
  return (ptr, len(v))

def store_record(opts, v, ptr, fields):
  for f in fields:
    ptr = align_to(ptr, alignment(f.t))
    store(opts, v[f.label], f.t, ptr)
    ptr += byte_size(f.t)

def store_variant(opts, v, ptr, cases):
  case_index, case_value = match_case(v, cases)
  disc_size = byte_size(discriminant_type(cases))
  store_int(opts, case_index, ptr, disc_size)
  ptr += disc_size
  ptr = align_to(ptr, max_alignment(types_of(cases)))
  store(opts, case_value, cases[case_index].t, ptr)

def match_case(v, cases):
  assert(len(v.keys()) == 1)
  key = list(v.keys())[0]
  value = list(v.values())[0]
  for label in key.split('|'):
    case_index = find_case(label, cases)
    if case_index != -1:
      return (case_index, value)
  assert_unreachable(key)

def store_flags(opts, v, ptr, labels):
  i = concat_flags_into_bigint(v, labels)
  store_int(opts, i, ptr, byte_size_flags(labels))

def concat_flags_into_bigint(v, labels):
  i = 0
  shift = 0
  for l in labels:
    i |= (int(bool(v[l])) << shift)
    shift += 1
  return i

# Flattening

def flatten(t):
  match despecialize(t):
    case Bool()               : return ['i32']
    case U8() | U16() | U32() : return ['i32']
    case S8() | S16() | S32() : return ['i32']
    case S64() | U64()        : return ['i64']
    case Float32()            : return ['f32']
    case Float64()            : return ['f64']
    case Char()               : return ['i32']
    case String() | List(_)   : return ['i32', 'i32']
    case Record(fields)       : return flatten_record(fields)
    case Variant(cases)       : return flatten_variant(cases)
    case Flags(labels)        : return flatten_flags(labels)
    case _                    : assert_unreachable(t)

def flatten_record(fields):
  return [flat_type for f in fields for flat_type in flatten(f.t)]

def flatten_variant(cases):
  flat = []
  for c in cases:
    for i, flat_type in enumerate(flatten(c.t)):
      if i < len(flat):
        flat[i] = join(flat[i], flat_type)
      else:
        flat.append(flat_type)
  return flatten(discriminant_type(cases)) + flat

def join(a, b):
  if a == b: return a
  if (a == 'i32' and b == 'f32') or (a == 'f32' and b == 'i32'): return 'i32'
  return 'i64'

def flatten_flags(labels):
  return ['i32'] * num_flattened_i32s(labels)

def num_flattened_i32s(labels):
  return math.ceil(len(labels) / 32)

# Lifting

@dataclass
class Value:
  t: any
  v: any

@dataclass
class ValueIter:
  values: [Value]
  i = 0
  def next(self, t):
    v = self.values[self.i]
    self.i += 1
    assert(v.t == t)
    return v.v

def lift(opts, vi, t):
  match despecialize(t):
    case Bool()         : return bool(vi.next('i32'))
    case U8()           : return lift_unsigned(vi, 32, 8)
    case U16()          : return lift_unsigned(vi, 32, 16)
    case U32()          : return lift_unsigned(vi, 32, 32)
    case U64()          : return lift_unsigned(vi, 64, 64)
    case S8()           : return lift_signed(vi, 32, 8)
    case S16()          : return lift_signed(vi, 32, 16)
    case S32()          : return lift_signed(vi, 32, 32)
    case S64()          : return lift_signed(vi, 64, 64)
    case Float32()      : return canonicalize(vi.next('f32'))
    case Float64()      : return canonicalize(vi.next('f64'))
    case Char()         : return i32_to_char(opts, vi.next('i32'))
    case String()       : return lift_string(opts, vi)
    case List(t)        : return lift_list(opts, vi, t)
    case Record(fields) : return lift_record(opts, vi, fields)
    case Variant(cases) : return lift_variant(opts, vi, cases)
    case Flags(labels)  : return lift_flags(vi, labels)
    case _              : assert_unreachable(t)

def lift_signed(vi, t_width, num_bits):
  i = vi.next('i' + str(t_width))
  assert(0 <= i < (1 << t_width))
  if i >= (1 << (num_bits - 1)):
    i -= (1 << t_width)
    trap_if(i < -(1 << (num_bits - 1)))
    return i
  trap_if(i >= (1 << (num_bits - 1)))
  return i

def lift_unsigned(vi, t_width, num_bits):
  i = vi.next('i' + str(t_width))
  assert(0 <= i < (1 << t_width))
  trap_if(i >= (1 << num_bits))
  return i

def lift_string(opts, vi):
  ptr = vi.next('i32')
  byte_length = vi.next('i32')
  return load_string_from_range(opts, ptr, byte_length)

def lift_list(opts, vi, elem_type):
  ptr = vi.next('i32')
  length = vi.next('i32')
  return load_list_from_range(opts, ptr, length, elem_type)

def lift_record(opts, vi, fields):
  record = {}
  for f in fields:
    record[f.label] = lift(opts, vi, f.t)
  return record

def lift_variant(opts, vi, cases):
  flat_types = flatten_variant(cases)
  i = vi.next(flat_types.pop(0))
  trap_if(i >= len(cases))
  case = cases[i]
  class CoerceValueIter:
    def next(self, want):
      have = flat_types.pop(0)
      x = vi.next(have)
      match (have, want):
        case ('i64', 'i32') : return reinterpret_i64_as_i32(x)
        case ('i32', 'f32') : return reinterpret_i32_as_float(x)
        case ('i64', 'f32') : return reinterpret_i32_as_float(reinterpret_i64_as_i32(x))
        case ('i64', 'f64') : return reinterpret_i64_as_float(x)
        case _              : return x
  v = lift(opts, CoerceValueIter(), case.t)
  for have in flat_types:
    _ = vi.next(have)
  return { case_label_with_defaults(case, cases): v }

def reinterpret_i64_as_i32(i):
  trap_if(i >= (1 << 32))
  return i

def lift_flags(vi, labels):
  i = 0
  shift = 0
  for _ in range(num_flattened_i32s(labels)):
    i |= (vi.next('i32') << shift)
    shift += 32
  return load_flags_from_bigint(i, labels)

# Lowering

def lower(opts, v, t):
  match despecialize(t):
    case Bool()         : return [Value('i32', int(v))]
    case U8()           : return [Value('i32', v)]
    case U16()          : return [Value('i32', v)]
    case U32()          : return [Value('i32', v)]
    case U64()          : return [Value('i64', v)]
    case S8()           : return lower_signed(v, 32)
    case S16()          : return lower_signed(v, 32)
    case S32()          : return lower_signed(v, 32)
    case S64()          : return lower_signed(v, 64)
    case Float32()      : return [Value('f32', v)]
    case Float64()      : return [Value('f64', v)]
    case Char()         : return [Value('i32', char_to_int(v))]
    case String()       : return lower_string(opts, v)
    case List(t)        : return lower_list(opts, v, t)
    case Record(fields) : return lower_record(opts, v, fields)
    case Variant(cases) : return lower_variant(opts, v, cases)
    case Flags(labels)  : return lower_flags(v, labels)
    case _              : assert_unreachable(t)

def lower_signed(i, num_bits):
  if i < 0:
    i += (1 << num_bits)
  return [Value('i' + str(num_bits), i)]

def lower_string(opts, v):
  ptr, byte_length = store_string_into_range(opts, v)
  return [Value('i32', ptr), Value('i32', byte_length)]

def lower_list(opts, v, elem_type):
  (ptr, length) = store_list_into_range(opts, v, elem_type)
  return [Value('i32', ptr), Value('i32', length)]
  
def lower_record(opts, v, fields):
  flat = []
  for f in fields: 
    flat += lower(opts, v[f.label], f.t)
  return flat

def lower_variant(opts, v, cases):
  case_index, case_value = match_case(v, cases)
  flat_types = flatten_variant(cases)
  assert(flat_types.pop(0) == 'i32')
  payload = lower(opts, case_value, cases[case_index].t)
  for i,have in enumerate(payload):
    want = flat_types.pop(0)
    match (have.t, want):
      case ('i32', 'i64') : payload[i] = Value('i64', have.v)
      case ('f32', 'i32') : payload[i] = Value('i32', reinterpret_float_as_i32(have.v))
      case ('f32', 'i64') : payload[i] = Value('i64', reinterpret_float_as_i32(have.v))
      case ('f64', 'i64') : payload[i] = Value('i64', reinterpret_float_as_i64(have.v))
      case _              : pass
  for want in flat_types:
    payload.append(Value(want, 0))
  return [Value('i32', case_index)] + payload

def lower_flags(v, labels):
  i = concat_flags_into_bigint(v, labels)
  flat = []
  for _ in range(num_flattened_i32s(labels)):
    flat.append(Value('i32', i & 0xffffffff))
    i >>= 32
  assert(i == 0)
  return flat

# Calling into a component

MAX_PARAMS = 16
MAX_RESULTS = 1 # only until we can use multi-value

class Instance:
  may_leave = True
  may_enter = True
  # ...

class FuncInst:
  instance: Instance
  func: any

def call_in(opts, callee, functype, args):
  trap_if(not callee.instance.may_enter)

  callee.instance.may_leave = False
  if len([t for p in functype.params for t in flatten(p)]) <= MAX_PARAMS:
    flat_args = []
    for i in range(len(functype.params)):
      flat_args += lower(opts, args[i], functype.params[i])
  else:
    heap_type = Tuple(functype.params)
    heap_value = {str(i): v for i,v in enumerate(args)}
    ptr = opts.realloc(0, 0, alignment(heap_type), byte_size(heap_type))
    store(opts, heap_value, heap_type, ptr)
    flat_args = [ Value('i32', ptr) ]
  callee.instance.may_leave = True

  flat_results = callee.func(flat_args)

  callee.instance.may_enter = False
  if len(flatten(functype.result)) <= MAX_RESULTS:
    result = lift(opts, ValueIter(flat_results), functype.result)
  else:
    result = load(opts, flat_results[0].v, functype.result)

  def post_return():
    callee.instance.may_enter = True
    opts.post_return()

  return (result, post_return)

# Calling out of a component

def call_out(opts, caller_instance, callee, functype, flat_args):
  trap_if(not caller_instance.may_leave)
  caller_instance.may_enter = False

  flat_args = ValueIter(flat_args)
  if len([t for p in functype.params for t in flatten(p)]) <= MAX_PARAMS:
    args = [ lift(opts, flat_args, p) for p in functype.params ]
  else:
    heap_type = Tuple(functype.params)
    heap_value = load(opts, flat_args.next('i32'), heap_type)
    args = list(heap_value.values())

  result, post_return = callee.func(args)

  caller_instance.may_leave = False
  if len(flatten(functype.result)) <= MAX_RESULTS:
    flat_results = lower(opts, result, functype.result)
  else:
    ptr = flat_args.next('i32')
    store(opts, result, functype.result, ptr)
    flat_results = [ Value('i32', ptr) ]
  caller_instance.may_leave = True

  post_return()

  caller_instance.may_enter = True
  return flat_results

