"""Read GGUF metadata and tensor offsets without mapping tensor data."""
import struct


def read_header(path):
    with open(path, 'rb') as f:
        def scalar(fmt):
            size = struct.calcsize('<' + fmt)
            data = f.read(size)
            if len(data) != size:
                raise EOFError('incomplete GGUF header')
            return struct.unpack('<' + fmt, data)[0]

        def string():
            length = scalar('Q')
            data = f.read(length)
            if len(data) != length:
                raise EOFError('incomplete GGUF string')
            return data.decode('utf-8')

        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
                   7: '?', 10: 'Q', 11: 'q', 12: 'd'}

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element_type, length = scalar('I'), scalar('Q')
                return [value(element_type) for _ in range(length)]
            return scalar(formats[kind])

        if f.read(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        version, tensor_count, kv_count = scalar('I'), scalar('Q'), scalar('Q')
        if version != 3:
            raise ValueError(f'unsupported GGUF version {version}')
        metadata = {}
        for _ in range(kv_count):
            key = string()
            metadata[key] = value(scalar('I'))
        tensors = {}
        for _ in range(tensor_count):
            name, dimensions = string(), scalar('I')
            shape = [scalar('Q') for _ in range(dimensions)]
            kind, offset = scalar('I'), scalar('Q')
            tensors[name] = {'shape': shape, 'type': kind, 'relative_offset': offset}
        alignment = metadata.get('general.alignment', 32)
        data_offset = (f.tell() + alignment - 1) // alignment * alignment
        for tensor in tensors.values():
            tensor['offset'] = data_offset + tensor['relative_offset']
        return {'version': version, 'metadata': metadata, 'data_offset': data_offset, 'tensors': tensors}
