// Stable LSD radix sort: eight nibbles of doc_id, then term_id nibbles.
// Each pass locally sorts 256 records, computes global bucket offsets,
// and scatters records. Original posting indices travel with their keys.
struct Params {
    count: u32,
    groups: u32,
    shift: u32,
    component: u32, // 1 = doc_id; 0 = term_id
}
@group(0) @binding(0) var<storage, read> input_records: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> output_records: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> local_records: array<vec4<u32>>;
// Bucket-major metadata: (count, local starting position).
@group(0) @binding(3) var<storage, read_write> histogram: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> scan: array<u32, 256>;
var<workgroup> records: array<vec4<u32>, 256>;
var<workgroup> totals: array<u32, 16>;

@compute @workgroup_size(256)
fn local_sort(@builtin(local_invocation_index) tid: u32,
              @builtin(workgroup_id) group: vec3<u32>) {
    let index = group.x * 256u + tid;
    let valid = min(256u, params.count - group.x * 256u);
    var record = vec4<u32>(0u);
    if tid < valid {
        record = input_records[index];
    }
    // Four stable binary partitions sort the current nibble locally.
    for (var bit = 0u; bit < 4u; bit++) {
        let zero = tid < valid && ((record[params.component] >> (params.shift + bit)) & 1u) == 0u;
        scan[tid] = select(0u, 1u, zero);
        workgroupBarrier();
        for (var stride = 1u; stride < 256u; stride *= 2u) {
            var add = 0u;
            if tid >= stride {
                add = scan[tid - stride];
            }
            workgroupBarrier();
            scan[tid] += add;
            workgroupBarrier();
        }
        let zeros = scan[255];
        let prefix = scan[tid];
        if tid < valid {
            var destination = zeros + tid - prefix;
            if zero {
                destination = prefix - 1u;
            }
            records[destination] = record;
        }
        workgroupBarrier();
        record = records[tid];
        workgroupBarrier();
    }
    if tid < valid {
        local_records[index] = record;
    }
    if tid < 16u {
        var count = 0u;
        var start = valid;
        for (var i = 0u; i < valid; i++) {
            if ((records[i][params.component] >> params.shift) & 15u) == tid {
                start = min(start, i);
                count++;
            }
        }
        histogram[tid * params.groups + group.x] = vec2<u32>(count, start);
    }
}

@compute @workgroup_size(16)
fn prefix(@builtin(local_invocation_index) bucket: u32) {
    var total = 0u;
    for (var group = 0u; group < params.groups; group++) {
        total += histogram[bucket * params.groups + group].x;
    }
    totals[bucket] = total;
    workgroupBarrier();
    var position = 0u;
    for (var earlier = 0u; earlier < bucket; earlier++) {
        position += totals[earlier];
    }
    for (var group = 0u; group < params.groups; group++) {
        let slot = bucket * params.groups + group;
        offsets[slot] = position;
        position += histogram[slot].x;
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) global: vec3<u32>) {
    let index = global.x;
    if index >= params.count {
        return;
    }
    let record = local_records[index];
    let bucket = (record[params.component] >> params.shift) & 15u;
    let slot = bucket * params.groups + index / 256u;
    let destination = offsets[slot] + index % 256u - histogram[slot].y;
    output_records[destination] = record;
}
