functions {
real mymetric(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("mymetric: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, ")");
    return ((x * x) + (y * y));
}
}
data {
    int x_n;
    vector[x_n] x;
    int y_n;
    vector[y_n] y;
}
transformed data {
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    real mu = std_normal_rng();
    real obs = mymetric((x + mu), y);
}