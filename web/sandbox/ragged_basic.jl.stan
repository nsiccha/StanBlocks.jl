functions {
int num_elements_RaggedVector(tuple(vector, array[] int) rv) {
    return size(rv.2);
}
int ragged_length_RaggedVector(
    tuple(vector, array[] int) x,
    int i
) {
    return ((ragged_end_RaggedVector(x, i) - ragged_start_RaggedVector(x, i)) + 1);
}
int ragged_end_RaggedVector(tuple(vector, array[] int) x, int i) {
    return x.2[i];
}
int ragged_start_RaggedVector(
    tuple(vector, array[] int) x,
    int i
) {
    if((i == 1)) {
        return 1;
    } else {
        return (1 + x.2[(i - 1)]);
    }
}
vector getindex_RaggedVector(
    tuple(vector, array[] int) rv,
    int i
) {
    return rv.1[ragged_start_RaggedVector(rv, i):ragged_end_RaggedVector(rv, i)];
}
}
data {
    int mem_n;
    int ends_n;
    vector[mem_n] mem;
    array[ends_n] int ends;
}
transformed data {
    tuple(vector[mem_n], array[ends_n] int) rv = (mem, ends);
    int n = num_elements_RaggedVector(rv);
    vector[ragged_length_RaggedVector(rv, 1)] g1 = getindex_RaggedVector(rv, 1);
    vector[ragged_length_RaggedVector(rv, 3)] g3 = getindex_RaggedVector(rv, 3);
    real tot = (sum(g1) + sum(g3));
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
}