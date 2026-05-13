functions {
}
data {
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
    real shift = std_normal_rng();
    real a = std_normal_rng();
    real b = (a + shift);
}