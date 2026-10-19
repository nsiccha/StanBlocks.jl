# A stand-in for BridgeStan's Makefile, for build-concurrency tests: the same
# shared-target variables and print-% rule, with recipes that only record what
# they observed. Each recipe appends "<target> locked|unlocked" to build.log by
# whether the toolchain lock file exists. A model recipe first waits, at most
# 30 s, until STANBLOCKS_TEST_RENDEZVOUS holds two entries, its own included,
# so two builds that cannot run at the same time fail instead of passing.
STANC ?= ./bin/stanc
BRIDGE_O = ./src/bridge$(if $(VARIANT),_$(VARIANT)).o
SUNDIALS_TARGETS = ./lib/sundials.a
MPI_TARGETS =
TBB_TARGETS = ./lib/tbb.so
SHARED = $(STANC) $(BRIDGE_O) $(SUNDIALS_TARGETS) $(MPI_TARGETS) $(TBB_TARGETS)
RENDEZVOUS = $(STANBLOCKS_TEST_RENDEZVOUS)
locked = $$(test -e .stanblocks-build.pid && echo locked || echo unlocked)

$(SHARED):
	@mkdir -p $(dir $@)
	@echo "$@ $(locked)" >> build.log
	@touch $@

%_model.so : %.stan $(SHARED)
	@touch "$(RENDEZVOUS)/$(notdir $*)"
	@n=0; until [ $$(ls "$(RENDEZVOUS)" | wc -l) -ge 2 ]; do n=$$((n+1)); [ $$n -lt 300 ] || exit 1; sleep 0.1; done
	@echo "model $(locked)" >> build.log
	@cp $< $@

print-% : ; @echo $* = $($*)
