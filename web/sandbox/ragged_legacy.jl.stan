functions {
int ragged_n(tuple(vector, array[] int) x) {
    return size(x.2);
}
int ragged_start(
    tuple(vector, array[] int) x,
    int i
) {
    if((i == 1)) {
        return 1;
    } else {
        return (1 + x.2[(i - 1)]);
    }
}
int ragged_end(tuple(vector, array[] int) x, int i) {
    return x.2[i];
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
    int n = ragged_n(rv);
    int s1 = ragged_start(rv, 1);
    int e1 = ragged_end(rv, 1);
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
}