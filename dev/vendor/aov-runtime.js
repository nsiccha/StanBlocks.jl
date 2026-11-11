window.AoV = window.AoV || {
    views: {},
    _pending: {},
    _signals: {},
    _liveRows: {},
    _specRows: {},
    _origSpecs: {},
    _droppedLegends: {},
    _embedOpts: {},
    _results: {},
    _els: {},
    _embedTok: {},
    _gens: {},
    _genSeq: 0,
    // The spec the current view was embedded from (resize/fit re-embeds
    // read it; update_spec swaps new data into it), the picker's current
    // channel-assignment builder, and the last applied remap mapping.
    _cur: {},
    _pickers: {},
    // config(frame=…) players: the frames, the shown frame, play state and
    // timer, and the controls element of each framed plot (see _frameSetup).
    _frames: {},
    _mappings: {},
    // The host-theme config each plot was last embedded with (JSON), and
    // the re-embed that applies a changed one (see refreshTheme).
    _themeKeys: {},
    _reembeds: {},
    // The tooltip looks written so far (look → theme name, see _tooltipTheme),
    // their stylesheet, and the tooltip handler for a cell chart per theme.
    _tipThemes: {},
    _tipStyle: null,
    _cellTips: {},
    // Keyed highlights (AoV.highlight): each plot's match test and the
    // revision its view's `aov_highlight` signal shows. Hovers: the mark
    // under the pointer, and the key it announced to its link group.
    _highlights: {},
    _hovers: {},
    // Touch input: the press in progress, the tooltip a tap left open, and
    // when the browser last emulated mouse events after one (see _attachTouch).
    _touches: {},
    // Container-sized plots last embedded while their container offered no
    // width (a hidden tab panel, a closed <details>; see _measurableAgain).
    _unmeasured: {},
    // Coloured ribbons drawn either as ribbons or as point-intervals: both
    // drawings, the one shown, and the overlap score (see setRibbonView).
    _ribbonViews: {},
    // The observers that keep a plot's scroll frame keyboard-reachable while
    // it scrolls (see _syncScrollFrame).
    _scrollFrames: {},

    // Every map keyed by plot id; dispose() clears each. A new per-plot
    // map belongs in this list.
    _perPlotState: ['views', '_results', '_els', '_embedTok', '_gens', '_pending',
        '_signals', '_liveRows', '_specRows', '_origSpecs', '_droppedLegends',
        '_embedOpts', '_observers', '_computedWidths', '_corrections', '_correctedRegime',
        '_cur', '_pickers', '_mappings', '_themeKeys', '_reembeds', '_frames',
        '_highlights', '_hovers', '_touches', '_unmeasured', '_ribbonViews', '_scrollFrames'],

    // Whether the runtime sizes this spec to its container: Julia's
    // `_embed_spec` marks a responsive layered/faceted/per-column spec with
    // `_aov.fitWidth`; a single view keeps VL's `width: "container"`, with or
    // without `_aov`. `_aov` alone is no sizing marker: it also carries
    // legend, theme and ribbon metadata, and a plot with an authored width
    // (`width=`, `fit_width=false`) keeps that width and is never re-embedded
    // on resize.
    _fitSized: function(spec) {
        return !!(spec && ((spec._aov && spec._aov.fitWidth) || spec.width === 'container'));
    },

    // A wrap facet (`layout=`) whose column count AoV chose itself: Julia's
    // `_apply_wrap_default!` writes AoG's near-square count and marks it
    // `_aov.autoColumns`. In the browser the row takes as many panels as fit the
    // plot's width instead (_wrapColumns), re-wrapping when that width changes.
    // An authored count (`config(columns=N)`, `wrapped(cols=N)`) carries no
    // marker and is never changed.
    _autoColumns: function(spec) {
        return !!(spec && spec._aov && spec._aov.autoColumns && this._wrapField(spec));
    },

    // The field of a wrap facet: the operator's `facet.field`, or a single
    // view's `encoding.facet.field`.
    _wrapField: function(spec) {
        if (spec.facet) return spec.facet.field || null;
        var f = spec.encoding && spec.encoding.facet;
        return (f && f.field) || null;
    },

    // How many distinct values `field` takes in `rows` (0 without rows).
    _distinctCount: function(rows, field) {
        if (!Array.isArray(rows) || !field) return 0;
        var seen = {};
        rows.forEach(function(r) { if (r && r[field] !== undefined) seen[r[field]] = true; });
        return Object.keys(seen).length;
    },

    // Panels per row for an auto-wrapped facet in `budget` px: each panel at its
    // authored width, or Vega-Lite's 200px default for a container-sized plot
    // (whose cells then stretch to fill the row), plus the facet spacing, ~40px
    // for the y axis on the left and ~6px of x-label overhang per panel
    // (measured on Vega-Lite 5.23: 170px panels take 196px each). At most one
    // row of every level; without inline rows the Julia count stays. A first
    // paint that still overshoots drops columns once (_fitCorrection).
    _wrapColumns: function(spec, budget, fit) {
        var n = this._distinctCount(spec.data && spec.data.values, this._wrapField(spec));
        if (!n) return spec.columns;
        var panel = spec.spec || spec;
        var cell = (!fit && typeof panel.width === 'number') ? panel.width : 200;
        var facetCfg = (spec.config && spec.config.facet) || {};
        var gap = typeof spec.spacing === 'number' ? spec.spacing :
            (typeof facetCfg.spacing === 'number' ? facetCfg.spacing : 20);
        var k = Math.floor((budget - 40 + gap) / (cell + gap + 6));
        return Math.max(1, Math.min(n, k));
    },

    // The width a JS-sized plot is laid out in: what its container offers,
    // in layout px (`vega_head(zoom=N)`), capped by `max_width`. `null` while
    // the container offers no width (a hidden tab panel; see _measurableAgain).
    _sizingBudget: function(el, spec) {
        var offered = this._offeredWidth(el);
        if (!offered || offered < 50) return null;
        var zoom = (window.AoV && window.AoV.zoom) ? window.AoV.zoom : 1;
        var maxWidth = (spec._aov && spec._aov.maxWidth) || (window.AoV && window.AoV.maxWidth) || Infinity;
        return Math.min(Math.floor(offered / zoom), maxWidth);
    },

    // The width el's container offers it, measured with el collapsed. A
    // container that shrink-wraps its content (an auto-layout table cell, an
    // inline-block, an auto grid/flex track) is widened by the plot it holds,
    // so measuring it with the plot in place feeds each render's overshoot into
    // the next one's budget. Collapsing only the width keeps the page height,
    // so the forced layout cannot clamp the scroll position.
    _offeredWidth: function(el) {
        var parent = el.parentElement;
        if (!parent) return null;
        var width = el.style.width;
        el.style.width = '0px';
        var offered = parent.clientWidth;
        el.style.width = width;
        return offered;
    },

    // A container-sized plot embedded while its container offered no width
    // rendered at its fallback width (layered/faceted) or 0px wide (a single
    // view: Vega-Lite reads `width: "container"` again only on a window
    // resize). Whatever lays the container out again — a tab panel shown on
    // load or by a click, a <details> opened — the plot re-sizes at once:
    // that is no drag-resize to coalesce, and the container may come back at
    // the width last seen before it was hidden. Measured as
    // _applyResponsiveWidth measures, so a container that still offers no
    // width (one that shrink-wraps the plot) re-embeds nothing.
    _measurableAgain: function(id, el) {
        if (!this._unmeasured[id]) return false;
        var offered = this._offeredWidth(el);
        if (!offered || offered < 50) return false;
        delete this._unmeasured[id];
        return true;
    },

    // `budget` pins the sizing budget instead of measuring the container: the
    // post-embed correction re-sizes for the budget it measured its overshoot
    // against (see _fitCorrection).
    _applyResponsiveWidth: function(id, spec, budget) {
        // A fixed-width or unmeasurable embed leaves no sizing record behind:
        // the post-embed correction must not resize it from a previous
        // responsive embed.
        var self = window.AoV || {};
        var unsized = function() {
            if (self._computedWidths) delete self._computedWidths[id];
            return spec;
        };
        var fit = this._fitSized(spec), wrap = this._autoColumns(spec);
        if (!fit && !wrap) return unsized();
        var el = document.getElementById(id);
        if (!el) return unsized();
        var containerWidth = budget;
        if (containerWidth === undefined) {
            containerWidth = this._sizingBudget(el, spec);
            if (containerWidth === null) {
                self._unmeasured[id] = true;
                return unsized();
            }
            delete self._unmeasured[id];
        }
        var padding = 30; // approximate VL padding
        var minCell = 100; // readable-minimum panel width; narrower viewports scroll in-frame (see .aov-plot-area)

        // An auto-wrapped facet first takes as many columns as fit; with
        // authored panel widths that is all the runtime changes.
        if (wrap) {
            var wrapEst = this._wrapColumns(spec, containerWidth, fit);
            var wrapRegime = [containerWidth, 'wrap', wrapEst, this._wrapField(spec)].join('|');
            var wrapCorr = (self._corrections || {})[id];
            var wrapCols = (wrapCorr && wrapCorr.regime === wrapRegime && wrapCorr.cols) ? wrapCorr.cols : wrapEst;
            spec = Object.assign({}, spec, {columns: wrapCols,
                _aov: Object.assign({}, spec._aov, {nFacetCols: wrapCols})});
            if (!fit) {
                self._computedWidths = self._computedWidths || {};
                self._computedWidths[id] = {kind: 'wrap', nCols: wrapCols, estimate: wrapEst,
                    budget: containerWidth, regime: wrapRegime};
                return spec;
            }
        }

        // Classify the shape. Only _aov-marked composite specs are JS-sized
        // (single views use VL-native width:"container").
        var kind = null, nCols = 0, computed = 0;
        if (spec._aov && spec._aov.nFacetCols && spec.hconcat) {
            // hconcat of per-column facet views (per-column Y scales):
            // per-child width from container / nCols, written into every
            // child's inner spec.
            kind = 'hconcat'; nCols = spec._aov.nFacetCols;
            computed = Math.max(minCell, Math.floor((containerWidth - padding) / nCols) - padding);
        } else if (spec._aov && spec._aov.nFacetCols && spec.spec) {
            // Operator-faceted specs: per-cell width from container / nCols
            kind = 'facet'; nCols = spec._aov.nFacetCols;
            computed = Math.max(minCell, Math.floor((containerWidth - padding) / nCols) - padding);
        } else if (spec._aov && spec._aov.nFacetCols && !spec.spec) {
            // Encoding-faceted unit specs: VL sizes top-level width PER
            // CELL here, so divide the same way but write it top-level.
            kind = 'encfacet'; nCols = spec._aov.nFacetCols;
            computed = Math.max(minCell, Math.floor((containerWidth - padding) / nCols) - padding);
        } else if (spec._aov && !spec._aov.nFacetCols) {
            // Row-only faceted (width lives on the inner spec) or layered
            // specs: one panel spans the container.
            computed = Math.max(minCell, containerWidth - padding);
            kind = spec.spec ? 'inner' : 'top';
        } else {
            // A single view without `_aov`: Vega-Lite reads its
            // `width: "container"` itself.
            return unsized();
        }
        var colField = spec.facet && spec.facet.column && spec.facet.column.field;
        var rowField = spec.facet && spec.facet.row && spec.facet.row.field;
        var regime = [containerWidth, kind, nCols, colField, rowField].join('|');

        // A post-embed correction (_fitCorrection) overrides the computed
        // width while its regime still matches; a resize/remap starts a
        // fresh regime and recomputes from the container.
        var corr = (self._corrections || {})[id];
        var width = (corr && corr.regime === regime) ? corr.width : computed;

        if (kind === 'hconcat') {
            spec.hconcat.forEach(function(child) {
                if (child && child.spec) child.spec = Object.assign({}, child.spec, {width: width});
            });
        } else if (kind === 'facet' || kind === 'inner') {
            spec.spec = Object.assign({}, spec.spec, {width: width});
        } else {
            // 'top' and 'encfacet' both write top-level width: total span
            // for plain/layered specs, per-cell width for encoding facets.
            spec = Object.assign({}, spec, {width: width});
        }
        self._computedWidths = self._computedWidths || {};
        self._computedWidths[id] = {kind: kind, width: width, computed: computed, nCols: nCols, budget: containerWidth, regime: regime};
        return spec;
    },

    // Post-embed correction for JS-sized specs. The computed cell width
    // reserves only approximate padding, while real facet chrome (row
    // headers, per-panel axes, spacing, legends) varies per spec — so the
    // first paint can overshoot the budget it was sized for. Measure the
    // rendered canvas; if it overshoots, shrink the cells by the excess
    // and re-embed once via reembed(budget). Bounded: at most one correction
    // per regime (budget/kind/columns/fields); a resize/remap opens a new
    // regime. The re-embed reuses the measured budget rather than measuring
    // again: in a container that grows with its content the overshooting
    // paint has already widened it, so a fresh measurement opened a new
    // regime on every pass and the corrections never ended. When cells
    // already sit at the readable minimum there is nothing to shrink — the
    // frame's in-frame scroll takes over.
    _fitCorrection: function(id, reembed) {
        var self = window.AoV || {};
        var info = (self._computedWidths || {})[id];
        if (!info) return;
        if (self._correctedRegime && self._correctedRegime[id] === info.regime) return;
        var el = document.getElementById(id);
        var canvas = el && el.querySelector('canvas.marks');
        if (!canvas) return;
        // Layout px (pre-zoom): CSS zoom scales the canvas and the budget's
        // container alike, so the factor cancels in the difference.
        var over = canvas.clientWidth - info.budget;
        self._correctedRegime = self._correctedRegime || {};
        self._corrections = self._corrections || {};
        var minCell = 100;
        var perCell = (info.kind === 'facet' || info.kind === 'encfacet' || info.kind === 'hconcat') && info.nCols > 0;
        // A relayout can also need less room than the correction made: a mark
        // at the domain's edge drawn past the frame for some rows, then not.
        // The plot widens back by what it falls short of its column, never
        // past the width computed for that column, so the correction follows
        // the chrome both ways instead of keeping the narrowest it ever had.
        // An empty plot does not widen: the room its missing tick labels free
        // comes back with its first rows, so widening it would only replace
        // the view twice (keyed removal of the last group, then its return).
        if (over < -2) {
            var corr = self._corrections[id];
            if (info.kind === 'wrap' || !corr || corr.regime !== info.regime || !(info.computed > info.width)) return;
            if (!self._drawsMarks(self.views[id])) return;
            var grow = perCell ? Math.floor(-over / info.nCols) : Math.floor(-over);
            var wider = Math.min(info.computed, info.width + grow);
            if (wider <= info.width) return;
            self._correctedRegime[id] = info.regime;
            self._corrections[id] = {regime: info.regime, width: wider};
            reembed(info.budget);
            return;
        }
        if (over <= 2) return;
        if (info.kind === 'wrap') {
            // Authored panel widths stay: the row takes fewer panels instead.
            if (info.nCols <= 1) return;
            var cols = Math.max(1, Math.min(info.nCols - 1,
                Math.floor(info.nCols * info.budget / canvas.clientWidth)));
            self._correctedRegime[id] = info.regime;
            self._corrections[id] = {regime: info.regime, cols: cols};
            reembed(info.budget);
            return;
        }
        var shrink = perCell ? Math.ceil(over / info.nCols) : Math.ceil(over);
        var newWidth = info.width - shrink;
        if (newWidth >= info.width || newWidth < minCell) return;
        self._correctedRegime[id] = info.regime;
        self._corrections[id] = {regime: info.regime, width: newWidth};
        reembed(info.budget);
    },

    // Whether a view draws any data mark item (axes, legends and titles aside).
    _drawsMarks: function(view) {
        var drawn = false;
        (function walk(mark) {
            if (drawn || !mark) return;
            if (mark.marktype === 'group') {
                (mark.items || []).forEach(function(g) { (g.items || []).forEach(walk); });
            } else if (mark.role === 'mark' && (mark.items || []).length) {
                drawn = true;
            }
        })(view && view.scenegraph().root);
        return drawn;
    },

    // Run a view whose data changed, laying it out again. Vega sizes the
    // view (autosize padding, and the plot width of an autosize "fit" view)
    // only when it is embedded or resized: its `autosize.resize` is false,
    // so a plain run keeps the layout of the rows it was embedded with. A
    // plot that starts empty has no axis labels then, and its axis titles
    // and the labels new rows bring end up outside the canvas. Only data
    // changes lay out again; pan and zoom keep the plot in place.
    _runRelayout: function(view) {
        view.resize().run();
    },

    // The relayout of a JS-sized plot can widen its chrome past the budget
    // it was sized for (an empty first paint has no axis labels). Vega calls
    // a view's resize listeners when it resizes the canvas, which after the
    // embed only a relayout does, and the overshoot is corrected as a first
    // paint's is, once for each data change that brings it.
    _refitOnRelayout: function(id, view, reembed) {
        var self = this;
        if (!(self._computedWidths || {})[id]) return;
        view.addResizeListener(function() {
            if (self.views[id] !== view) return;
            if (self._correctedRegime) delete self._correctedRegime[id];
            self._fitCorrection(id, reembed);
        });
    },

    // ggplot-style "broadcast across all facet panels" pass.
    // AoV merges layered data into a single spec.data.values with a __src
    // discriminator column when outer faceting is needed. When a facet field
    // is absent from some rows (e.g. dose VLines that have no health column),
    // Vega-Lite renders them in an "undefined" facet panel.
    // This pass replicates each missing-field row across the unique values
    // of that field. Layer-level __src filters still route rows correctly.
    // Idempotent: rows that already have the field are untouched.
    _broadcastCrossSource: function(spec) {
        if (!spec || typeof spec !== 'object') return spec;
        var inner = (spec.spec && typeof spec.spec === 'object') ? spec.spec : null;

        // Collect partition fields from facet config AND sublayer
        // row/column encodings. Deliberately NOT color: a missing color
        // value renders as one null group, never an "undefined" panel,
        // so replicating rows across color values only overpaints
        // identical geometry.
        var fields = [];
        var pushField = function(f) {
            if (f && f.field && fields.indexOf(f.field) === -1) fields.push(f.field);
        };
        if (spec.facet) { pushField(spec.facet.row); pushField(spec.facet.column); }
        if (spec.hconcat) {
            spec.hconcat.forEach(function(child) {
                if (!child) return;
                if (child.facet) { pushField(child.facet.row); pushField(child.facet.column); }
                var cl = child.spec && child.spec.layer ? child.spec.layer : null;
                if (cl) cl.forEach(function(l) {
                    if (l && l.encoding) { pushField(l.encoding.row); pushField(l.encoding.column); }
                });
            });
        }
        var layers = inner && inner.layer ? inner.layer : (spec.layer || null);
        if (layers) {
            layers.forEach(function(l) {
                if (l && l.encoding) { pushField(l.encoding.row); pushField(l.encoding.column); }
            });
        }
        if (!fields.length) return spec;

        // Find the data object whose values carry the merged rows
        var dataObj = null;
        if (spec.data && spec.data.values) dataObj = spec.data;
        else if (inner && inner.data && inner.data.values) dataObj = inner.data;
        if (!dataObj) return spec;
        var vals = dataObj.values;

        fields.forEach(function(field) {
            var uniques = [];
            var seen = {};
            for (var i = 0; i < vals.length; i++) {
                var v = vals[i];
                if (v && Object.prototype.hasOwnProperty.call(v, field)) {
                    var k = String(v[field]);
                    if (!seen[k]) { seen[k] = true; uniques.push(v[field]); }
                }
            }
            if (!uniques.length) return;

            var newVals = [];
            for (var i = 0; i < vals.length; i++) {
                var v = vals[i];
                if (!v || Object.prototype.hasOwnProperty.call(v, field)) {
                    newVals.push(v);
                } else {
                    for (var j = 0; j < uniques.length; j++) {
                        var nv = Object.assign({}, v);
                        nv[field] = uniques[j];
                        newVals.push(nv);
                    }
                }
            }
            vals = newVals;
        });
        dataObj.values = vals;

        return spec;
    },

    embed: function(id, spec, opts) {
        return this._embed(id, this._takeRibbonViews(id, spec, false), opts, false);
    },

    // --- Ribbons | Intervals ---
    // A coloured ribbon over a few x values arrives with two drawings
    // (src/ribbon_view.jl): the spec shows the starting one and carries the
    // other as `_aov.ribbonView.alt`. Record both and return the drawing to
    // show: the starting one, or, for a new spec of the same plot (`keep`,
    // update_spec), the one the reader chose. The drawings are stored without
    // that metadata, so an update that changes only data still swaps in place.
    _takeRibbonViews: function(id, spec, keep) {
        var rv = spec && spec._aov && spec._aov.ribbonView;
        var prev = this._ribbonViews[id];
        if (!rv || !rv.alt) {
            delete this._ribbonViews[id];
            return rv ? this._withoutRibbonView(spec) : spec;
        }
        var specs = {};
        specs[rv.view] = this._withoutRibbonView(spec);
        specs[rv.alt._aov.ribbonView.view] = this._withoutRibbonView(rv.alt);
        var view = keep && prev && specs[prev.view] ? prev.view : rv.view;
        this._ribbonViews[id] = {view: view, specs: specs, score: rv.score, threshold: rv.threshold};
        return specs[view];
    },
    _withoutRibbonView: function(spec) {
        var out = Object.assign({}, spec), aov = Object.assign({}, spec._aov);
        delete aov.ribbonView;
        if (Object.keys(aov).length) out._aov = aov; else delete out._aov;
        return out;
    },

    // Public: show plot `id` as 'ribbons' or 'intervals' (its "Ribbons |
    // Intervals" switch). The picker's channels, rows added since the first
    // render and the host theme carry over, as on a picker change. Returns
    // whether the plot has that drawing.
    setRibbonView: function(id, view) {
        var st = this._ribbonViews[id];
        if (!st || !st.specs[view]) return false;
        if (st.view === view) return true;
        st.view = view;
        var pristine = this._expandSpecData(JSON.parse(JSON.stringify(st.specs[view])));
        this._refreshLegendSelections(pristine);
        this._broadcastCrossSource(pristine);
        var shown = pristine;
        if (this._mappings[id]) {
            // The picker builds its spec from the stored original.
            this._origSpecs[id] = pristine;
            var mapping = this._pickers[id] ? this._pickers[id]() : this._mappings[id];
            shown = (mapping && this._remappedSpec(id, mapping)) || pristine;
        }
        var live = (this._liveRows[id] || {}).source_0;
        if (live) shown = this._withRibbonGroups(shown, live) || shown;
        this._embed(id, shown, this._embedOpts[id], true);
        this._origSpecs[id] = pristine;
        this._dataChanged(id);
        return true;
    },

    // The switch of a plot with two drawings: a pressed/unpressed pair of
    // buttons, first in its captioned figure's action row (HTMXObjects'
    // caption styles give them the row's look), else in a group right after
    // the plot (as a frame player). Removed when a new spec no longer
    // offers the point-interval drawing.
    _ribbonViewControls: function(id, el) {
        var self = this, st = this._ribbonViews[id];
        var sel = '[data-aov-ribbon-view-for="' + id + '"]';
        var buttons = document.querySelectorAll(sel);
        if (!st) {
            buttons.forEach(function(b) {
                var group = b.parentElement;
                b.remove();
                if (group && group.classList.contains('aov-ribbon-view') && !group.children.length) group.remove();
            });
            return;
        }
        if (!buttons.length && el && el.parentNode) {
            var fig = el.closest('figure.captioned');
            var acts = fig && fig.querySelector('.caption-actions');
            var header = fig && fig.querySelector('.caption-header');
            if (!acts && header) {
                acts = document.createElement('span');
                acts.className = 'caption-actions';
                header.appendChild(acts);
            }
            var made = ['ribbons', 'intervals'].map(function(v) {
                var b = document.createElement('button');
                b.type = 'button';
                b.className = acts ? 'outline caption-action' : 'outline';
                b.dataset.view = v;
                b.setAttribute('data-aov-ribbon-view-for', id);
                b.addEventListener('click', function() { self.setRibbonView(id, v); });
                return b;
            });
            if (acts) {
                acts.insertBefore(made[1], acts.firstChild);
                acts.insertBefore(made[0], made[1]);
            } else {
                var group = document.createElement('div');
                group.className = 'aov-ribbon-view';
                group.setAttribute('role', 'group');
                made.forEach(function(b) { group.appendChild(b); });
                var host = el.parentElement && el.parentElement.classList.contains('aov-plot-area') ? el.parentElement : el;
                host.insertAdjacentElement('afterend', group);
            }
            buttons = document.querySelectorAll(sel);
        }
        if (!buttons.length) return;
        var text = this._texts(buttons[0]);
        var pct = function(x) { return Math.round(100 * x) + '%'; };
        var tip = st.threshold === null || st.threshold === undefined ?
            text('overlap', {score: pct(st.score)}) :
            text('overlap_auto', {score: pct(st.score), threshold: pct(st.threshold)});
        var group = buttons[0].parentElement;
        if (group && group.classList.contains('aov-ribbon-view')) group.setAttribute('aria-label', text('ribbon_view'));
        buttons.forEach(function(b) {
            b.textContent = text(b.dataset.view);
            b.title = tip;
            b.setAttribute('aria-pressed', String(b.dataset.view === st.view));
        });
    },

    // --- Polygon winding (geometry loaded from a URL) ---
    // d3-geo reads a counter-clockwise exterior ring (RFC 7946's winding) as
    // the whole globe minus the polygon, so such a map fills the projection
    // in one colour. AoV draws each polygon as the smaller region its rings
    // enclose: an exterior ring covering more than a hemisphere is reversed,
    // and so is a hole that does not. Julia rewinds inline geometry when it
    // lowers a geoshape; a URL is fetched here, so every plot's loader
    // rewinds the GeoJSON it reads. LOCKSTEP: `_rewind_geojson!` and
    // `_ring_area` in src/analysis_to_vl.jl (snag `a-choropleth-of-89e8fdc1`).

    // d3-geo's area of a polygon with this one ring, in steradians (0 to 4π).
    _ringArea: function(ring) {
        var rad = Math.PI / 180, n = ring.length - 1, p = ring[0];
        var l0 = p[0] * rad, f = p[1] * rad / 2 + Math.PI / 4;
        var c0 = Math.cos(f), s0 = Math.sin(f), sum = 0;
        for (var i = 1; i <= n; i++) {
            p = i < n ? ring[i] : ring[0];
            var l = p[0] * rad, d = l - l0, sd = d >= 0 ? 1 : -1;
            f = p[1] * rad / 2 + Math.PI / 4;
            var c = Math.cos(f), s = Math.sin(f), k = s0 * s;
            sum += Math.atan2(k * sd * Math.sin(sd * d), c0 * c + k * Math.cos(sd * d));
            l0 = l; c0 = c; s0 = s;
        }
        return 2 * (sum < 0 ? 2 * Math.PI + sum : sum);
    },
    // Rewinds a parsed GeoJSON object in place; returns whether a ring changed.
    _rewindGeoJSON: function(x) {
        var self = this, any = function(xs, f) {
            return Array.isArray(xs) && xs.reduce(function(ch, e, i) { return f(e, i) || ch; }, false);
        };
        if (Array.isArray(x)) return any(x, function(e) { return self._rewindGeoJSON(e); });
        if (!x || typeof x !== 'object') return false;
        var polygon = function(rings) {
            return any(rings, function(ring, i) {
                if (!Array.isArray(ring) || ring.length < 4) return false;
                // The exterior ring (the first) must cover less than a hemisphere, a hole more.
                if ((self._ringArea(ring) > 2 * Math.PI) !== (i === 0)) return false;
                ring.reverse();
                return true;
            });
        };
        switch (x.type) {
            case 'FeatureCollection': return self._rewindGeoJSON(x.features);
            case 'Feature': return self._rewindGeoJSON(x.geometry);
            case 'GeometryCollection': return self._rewindGeoJSON(x.geometries);
            case 'Polygon': return polygon(x.coordinates);
            case 'MultiPolygon': return any(x.coordinates, function(rings) { return polygon(rings); });
            default: return false;
        }
    },
    // What a data URL's load returned, with any GeoJSON polygons rewound: a
    // JSON dataset arrives parsed (Vega asks the fetch for JSON), a file read
    // or another format as text. Text with no polygon type, and text that is
    // not JSON (a CSV, say), comes back as it was, for Vega to read in the
    // dataset's own format.
    _rewindLoaded: function(loaded) {
        if (loaded && typeof loaded === 'object') { this._rewindGeoJSON(loaded); return loaded; }
        if (typeof loaded !== 'string' || !/"(Multi)?Polygon"/.test(loaded)) return loaded;
        var obj;
        try { obj = JSON.parse(loaded); } catch (e) { return loaded; }
        return this._rewindGeoJSON(obj) ? JSON.stringify(obj) : loaded;
    },
    // The embed options with a loader that rewinds what it loads: the page's
    // own loader (a Vega loader, or loader options) or Vega's default. A data
    // URL a saved card carries (`_preload`) is answered from the card itself.
    _withGeoLoader: function(opts) {
        var self = this, given = opts.loader;
        var base = given && typeof given.load === 'function' ? given : vega.loader(given);
        var loader = Object.create(base);
        loader.load = function(uri, options) {
            var loading = Object.prototype.hasOwnProperty.call(self._preloaded, uri) ?
                Promise.resolve(self._preloaded[uri]) : base.load.call(this, uri, options);
            return loading.then(function(loaded) { return self._rewindLoaded(loaded); });
        };
        return Object.assign({}, opts, {loader: loader});
    },

    // Data URL → the text it answered, for a page with no server to ask: a
    // "⬇ HTML" card writes the files its plot loads (a `by_url` table, a
    // `geoshape(url)` geometry) into one `AoV._preload({...})` call.
    _preloaded: {},
    _preload: function(texts) {
        Object.assign(this._preloaded, texts);
    },
    // The data URLs a spec loads: every `data` object naming a `url`,
    // including a geoshape lookup's `from.data`.
    _dataUrls: function(spec) {
        var urls = [];
        (function walk(x) {
            if (Array.isArray(x)) { x.forEach(walk); return; }
            if (!x || typeof x !== 'object') return;
            Object.keys(x).forEach(function(k) {
                var v = x[k];
                if (k === 'data' && v && typeof v.url === 'string' && urls.indexOf(v.url) === -1) urls.push(v.url);
                walk(v);
            });
        })(spec);
        return urls;
    },
    // The `<script>` a saved card carries for its plot's data URLs: each
    // file's text, from this page's preload when it has one (a card saved
    // from a saved card), else fetched as the plot loaded it.
    _preloadScript: function(id) {
        var self = this, cur = self._cur[id];
        var urls = cur ? self._dataUrls(cur.spec) : [];
        if (!urls.length) return Promise.resolve('');
        return Promise.all(urls.map(function(url) {
            if (Object.prototype.hasOwnProperty.call(self._preloaded, url)) return self._preloaded[url];
            return fetch(url).then(function(r) {
                if (!r.ok) throw new Error('AoV.downloadPlotHtml: ' + url + ' answered HTTP ' + r.status);
                return r.text();
            });
        })).then(function(texts) {
            var map = {};
            urls.forEach(function(url, i) { map[url] = texts[i]; });
            // The same closing-tag rule as `_inlineVendored`.
            var json = JSON.stringify(map).replace(/<\/(?=(?:script|style|head|body|html)\b)/gi, '<\\/');
            return '<script>AoV._preload(' + json + ');<\/script>';
        });
    },

    // A plot whose element leaves the document is disposed. One
    // MutationObserver (installed on the first embed) sweeps the registry
    // after any batch of DOM removals. Its callback runs at the next
    // microtask checkpoint, so a node removed and re-inserted synchronously
    // stays live; it compares element identity, so a same-id re-render
    // already embedded into its new element is untouched.
    _sweepDetached: function() {
        var self = this;
        Object.keys(self._els).forEach(function(id) {
            var el = self._els[id];
            if (el && !el.isConnected) self.dispose(id);
        });
        self._sweepCells();
    },
    _watchRemovals: function() {
        if (this._removalObserver || typeof MutationObserver === 'undefined' || !document.documentElement) return;
        var self = this;
        this._removalObserver = new MutationObserver(function(records) {
            for (var i = 0; i < records.length; i++) {
                if (records[i].removedNodes.length) { self._sweepDetached(); return; }
            }
        });
        this._removalObserver.observe(document.documentElement, {childList: true, subtree: true});
    },

    // The primitive units of a spec, each with the data and encoding it
    // inherits from its ancestors and its colour-legend scope. Vega-Lite merges
    // the legends of a shared colour scale whatever fields its units colour
    // by; an independent colour scale or legend gives each child its own.
    _legendUnits: function(spec) {
        var units = [], scopes = 1;
        function independent(node) {
            var r = node.resolve || {};
            return ['scale', 'legend'].some(function(k) { return !!r[k] && r[k].color === 'independent'; });
        }
        function visit(node, data, encoding, scope) {
            if (!node) return;
            data = node.data || data;
            encoding = Object.assign({}, encoding, node.encoding || {});
            var children = node.spec ? [node.spec] : [];
            ['layer', 'hconcat', 'vconcat', 'concat'].forEach(function(k) {
                children = children.concat(node[k] || []);
            });
            var split = independent(node);
            if (children.length) children.forEach(function(c) { visit(c, data, encoding, split ? ++scopes : scope); });
            else if (node.mark) units.push({unit:node, data:data, encoding:encoding, scope:scope});
        }
        visit(spec, null, {}, 1);
        return units;
    },
    // Undo what _refreshLegendSelections added to a spec's units.
    _stripLegendSelections: function(spec) {
        this._legendUnits(spec).forEach(function(info) {
            var u = info.unit, meta = u._aovLegend;
            if (!meta) return;
            if (u.params) {
                u.params = u.params.filter(function(p) { return meta.params.indexOf(p.name) === -1; });
                if (!u.params.length) delete u.params;
            }
            if (meta.dimmed) {
                if (meta.opacity === null) delete u.encoding.opacity;
                else u.encoding.opacity = meta.opacity;
            }
            if (meta.domain && u.encoding.color && u.encoding.color.scale &&
                JSON.stringify(u.encoding.color.scale.domain) === JSON.stringify(meta.domain)) {
                delete u.encoding.color.scale.domain;
                if (!Object.keys(u.encoding.color.scale).length) delete u.encoding.color.scale;
            }
            if (meta.filtered) {
                if (meta.transform.length) u.transform = meta.transform;
                else delete u.transform;
            }
            delete u._aovLegend;
        });
    },
    // Refresh ONLY AoV-generated legend selections after channel remapping.
    // Units may have been cloned by ribbon reconstruction or facet/concat
    // lowering. Restore their recorded opacity/filter before electing one
    // owner per colour legend; explicit user params and encodings survive
    // unchanged. Mirrors `_add_auto_legend_interactivity!` (to_vegalite.jl).
    _refreshLegendSelections: function(spec) {
        var mode = spec._aov && spec._aov.legendInteraction;
        if (!mode) return;
        this._stripLegendSelections(spec);
        var units = this._legendUnits(spec);
        var legends = Object.create(null), order = [];
        function composite(u) {
            var m = typeof u.mark === 'string' ? u.mark : u.mark.type;
            return ['boxplot', 'errorbar', 'errorband'].indexOf(m) !== -1;
        }
        function legendField(info) {
            var c = info.encoding.color;
            if (composite(info.unit) || !c || !c.field ||
                ['nominal', 'ordinal'].indexOf(c.type) === -1 || c.legend === null || c.scale === null ||
                c.aggregate || c.bin || c.timeUnit) return null;
            return String(c.field);
        }
        // Vega-Lite names a merged legend after its first field, so each
        // legend's selection projects that field, owned by its first unit.
        units.forEach(function(info) {
            var f = legendField(info);
            if (f === null) return;
            var legend = legends[info.scope];
            if (!legend) { legend = legends[info.scope] = {owner:info.unit, fields:[]}; order.push(info.scope); }
            if (legend.fields.indexOf(f) === -1) legend.fields.push(f);
        });
        function selName(s) {
            var i = order.indexOf(s);
            return i ? 'legend_selection_' + (i+1) : 'legend_selection';
        }
        // Clicked entries are stored under the legend's bound field; another
        // field of a merged legend uses Vega-Lite's own legend-entry test.
        function legendTest(s, tested) {
            var name = selName(s), bound = legends[s].fields[0];
            var valid = tested.map(function(f) { return 'isValid(datum[' + JSON.stringify(f) + '])'; });
            var selected = name + '[' + JSON.stringify(bound) + ']';
            return {or:[valid.length === 1 ? '!' + valid[0] : '!(' + valid.join(' || ') + ')'].concat(
                tested.map(function(f) {
                    return f === bound ? {param:name, empty:true} :
                        '!length(data(' + JSON.stringify(name + '_store') + ')) || (' + selected +
                        ' && indexof(' + selected + ', datum[' + JSON.stringify(f) + ']) >= 0)';
                }))};
        }
        function hasField(v, field) {
            return v && typeof v === 'object' && (v.field === field ||
                Object.keys(v).some(function(k) { return hasField(v[k], field); }));
        }
        units.forEach(function(info) {
            var u = info.unit;
            if (composite(u)) return;
            var rows = info.data && info.data.values || [];
            function carries(f) {
                return hasField(info.encoding, f) || rows.some(function(r) {
                    return Object.prototype.hasOwnProperty.call(r, f);
                });
            }
            // A unit coloured by a legend's field answers to that field alone.
            var own = legendField(info);
            var tests = [];
            order.forEach(function(s) {
                var fields = legends[s].fields;
                var tested = info.scope === s && fields.indexOf(own) !== -1 ? [own] : fields.filter(carries);
                if (tested.length) tests.push({scope:s, fields:tested});
            });
            if (!tests.length) return;
            var enc = Object.assign({}, info.encoding);
            u.encoding = enc;
            var meta = {params:[], dimmed:false, filtered:false,
                opacity:enc.opacity === undefined ? null : enc.opacity,
                transform:u.transform || []};
            u._aovLegend = meta;
            var color = enc.color;
            if (mode === 'filter' && color && color.field && color.scale !== null &&
                !(color.scale && 'domain' in color.scale)) {
                var domain = [];
                units.forEach(function(peer) {
                    (peer.data && peer.data.values || []).forEach(function(row) {
                        var value = row[color.field];
                        if (value !== undefined && value !== null && domain.indexOf(value) === -1) domain.push(value);
                    });
                });
                if (domain.length) {
                    enc.color = Object.assign({}, color, {scale:Object.assign({}, color.scale, {domain:{unionWith:domain}})});
                    meta.domain = {unionWith:domain};
                }
            }
            tests.forEach(function(t) {
                if (legends[t.scope].owner !== u) return;
                var name = selName(t.scope);
                u.params = (u.params || []).concat([{name:name,
                    select:{type:'point', fields:[legends[t.scope].fields[0]]}, bind:'legend'}]);
                meta.params.push(name);
            });
            var predicate = {and:tests.map(function(t) { return legendTest(t.scope, t.fields); })};
            if (mode === 'filter') {
                meta.filtered = true;
                u.transform = meta.transform.concat([{filter:predicate}]);
            } else {
                var base = enc.opacity === undefined ?
                    (typeof u.mark === 'object' && u.mark.opacity !== undefined ? u.mark.opacity : 1) :
                    enc.opacity && enc.opacity.value;
                if (typeof base === 'number' && !(enc.opacity && enc.opacity.condition)) {
                    meta.dimmed = true;
                    enc.opacity = {condition:{test:predicate, value:base}, value:0.15*base};
                }
            }
        });
    },

    // `keepData` retains rows added via updateData/appendData on re-embed.
    _embed: function(id, spec, opts, keepData) {
        opts = opts || {};
        if (window.AoV && window.AoV.defaultActions !== undefined) {
            opts = Object.assign({}, opts, {actions: window.AoV.defaultActions});
        }
        var self = this;
        self._watchRemovals();
        self._watchTheme();
        self._watchLinks();
        // Live state (rows added via updateData/appendData, onSignal listeners) belongs
        // to the plot element: a new element with this ID (e.g. after an HTMX swap)
        // starts fresh, a new spec for the same element keeps the listeners.
        // Same element: the one the last embed recorded — which also covers
        // a re-embed issued while the previous one is still in flight (no
        // view registered yet), e.g. the two remaps of one picker change.
        var prev = self.views[id], el = document.getElementById(id);
        var sameEl = el && (self._els[id] === el || (prev && el.contains(prev.container())));
        if (!sameEl) {
            // A hover on the replaced element ends (its link group hears the
            // key go), and a highlight set on it does not carry over.
            self._touchReset(id);
            self._hoverEnd(id, null);
            delete self._highlights[id];
            delete self._signals[id];
            delete self._liveRows[id];
            delete self._droppedLegends[id];
            delete self._mappings[id];
            self._frameStop(id);
            delete self._frames[id];
        } else if (!keepData) {
            delete self._liveRows[id];
        }
        // A new spec is sized afresh, on a new element or the same one (a
        // picker remap, a structural update_spec, a legend restore). The fit
        // correction the previous spec earned narrowed it for that spec's
        // chrome, and its sizing regime (budget, shape, columns, facet
        // fields) is usually this spec's too: kept, it sized a plot that
        // gained a legend or wider axis labels for the old chrome, and
        // _fitCorrection then declined to correct the regime again, so the
        // legend lay past the frame. Resize and correction re-embeds go
        // through doEmbed, not here, and keep it.
        if (self._corrections) delete self._corrections[id];
        if (self._correctedRegime) delete self._correctedRegime[id];
        // The element this plot lives in (disposeWithin matches on it) and
        // this embed's ownership token: resize/fit re-embeds scheduled by an
        // earlier embed of the id check it and do nothing once superseded.
        var tok = {};
        self._els[id] = el;
        self._embedTok[id] = tok;
        // Remember the embed options so the restore re-embed below (and any
        // legend-restore re-embed from appendData/updateData) can preserve
        // them (e.g. actions:false) instead of falling back to defaults.
        self._embedOpts[id] = opts;
        // GeoJSON a data URL returns is rewound before Vega reads it, in the
        // view and in the frame player's headless one.
        opts = self._withGeoLoader(opts);
        // Store original spec for re-embed on resize and remapEncoding
        var origSpec = self._expandSpecData(JSON.parse(JSON.stringify(spec)));
        self._refreshLegendSelections(origSpec);
        spec = origSpec;
        self._broadcastCrossSource(origSpec);
        self._origSpecs[id] = origSpec;
        // Held by reference so an in-place data swap (updateSpec) reaches
        // the resize/fit re-embeds this embed schedules.
        var cur = self._cur[id] = {spec: origSpec};
        // A linked plot that appears while its group shows a key (a fragment
        // swapped in under a hovered table row) shows it too.
        if (!sameEl) self._joinLinkState(id);
        // config(frame=…): the player and its controls, and the scale domains
        // every frame shares, computed once per spec before the first embed.
        var framePlan = self._frameSetup(id, el, origSpec, opts);
        self._ribbonViewControls(id, el);

        // width:"container" sizes the view from the embed element's own
        // width, and vega-embed makes that element display:inline-block,
        // which shrink-wraps its (not yet rendered) content: unless a page
        // stylesheet happens to widen it, the container measures 0 and the
        // plot renders 0px wide. Size the element here, where the runtime
        // runs, so single-view plots fill their container on any page
        // (vega_head pages, to_html files, runtime-only docs embeds).
        if (spec.width === 'container' && el) el.style.width = '100%';

        // Width cap for VL-native single-view specs: width:"container" has
        // no max, so bound the embed element itself and the plot fills
        // min(container, cap). (JS-sized layered/faceted specs are capped
        // through the sizing budget in _applyResponsiveWidth instead.)
        var _cap = (spec._aov && spec._aov.maxWidth) || (window.AoV && window.AoV.maxWidth);
        if (_cap !== undefined && _cap !== null && isFinite(_cap) && spec.width === 'container') {
            var _capEl = document.getElementById(id);
            if (_capEl) _capEl.style.maxWidth = _cap + 'px';
        }

        // `budget` (the correction re-embed only) pins the sizing budget.
        var doEmbed = function(budget) {
            if (!framePlan) return embedNow(budget, null);
            return framePlan.then(function(plan) { return embedNow(budget, plan); });
        };
        var embedNow = function(budget, plan) {
            // Superseded by a newer embed of this id, or disposed: a resize or
            // fit re-embed scheduled earlier must not resurrect the old spec.
            if (self._embedTok[id] !== tok) return Promise.resolve();
            var s = self._withHighlight(id, self._applyResponsiveWidth(id, JSON.parse(JSON.stringify(cur.spec)), budget));
            // Host theme, read now: a theme change re-runs this embed.
            var embedOpts = self._withFrame(id, self._themedOpts(id, s, opts), plan);
            var fonts = self._fontsPending((s.config && s.config.font) || (embedOpts.config && embedOpts.config.font));
            return fonts ? fonts.then(function() { return render(s, embedOpts); }) : render(s, embedOpts);
        };
        var render = function(s, embedOpts) {
            // Superseded while its font loaded.
            if (self._embedTok[id] !== tok) return Promise.resolve();
            // Replace (not leak) the previous view
            self._finalizeView(id);
            // Only the latest embed registers its view; one that resolves after
            // a newer embed (or a dispose) started is finalized instead.
            var gen = self._gens[id] = ++self._genSeq;
            // Tag VL warnings/errors with the plot ID for easier debugging
            self._tagConsole();
            self._inFlight[id] = (self._inFlight[id] || 0) + 1;
            var settled = false;
            var settle = function() {
                if (settled) return;
                settled = true;
                if (--self._inFlight[id] <= 0) delete self._inFlight[id];
            };
            return vegaEmbed('#' + id, s, self._withLiveRows(id, embedOpts, gen)).then(function(result) {
                settle();
                if (self._gens[id] !== gen) {
                    result.finalize();
                    // Disposed while in flight, and no newer embed owns the
                    // element: this embed's drawing stays, so release it as
                    // dispose does. A newer embed of the element redraws it.
                    if (self._els[id] !== el) self._releaseDrawing(id, el);
                    return result;
                }
                self._results[id] = result;
                var view = self.views[id] = result.view;
                (self._signals[id] || []).forEach(function(sig) {
                    self._attachSignal(view, sig.signal, sig.callback);
                });
                self._attachClick(id, view);
                self._attachHover(id, view);
                self._attachTouch(id, view);
                self._frameSync(id, view);
                // Run anything queued before the view was ready (e.g. appendData)
                var pending = self._pending[id] || [];
                delete self._pending[id];
                pending.forEach(function(fn) { fn(view); });
                // One bounded correction pass for JS-sized specs whose
                // rendered chrome overshoots the computed budget. Not for an
                // embed a newer spec has superseded before it resolved: its
                // correction would describe the old chrome and be read by
                // the newer spec's first render.
                if (self._embedTok[id] === tok) self._fitCorrection(id, function(budget) { doEmbed(budget); });
                self._refitOnRelayout(id, view, function(budget) { doEmbed(budget); });
                self._watchScrollFrame(id);
                return result;
            }).catch(function(err) { settle(); self._console.error.call(console, '[' + id + ']', err); });
        };

        // A host-theme change re-embeds like a resize does: the current
        // spec (picker assignment, update_spec data) and live rows carry over.
        self._reembeds[id] = doEmbed;
        // The first render waits for the parsed page (_afterParse), and the
        // resize observer starts from the width that render is sized for.
        return self._afterParse(function() {
            if (self._embedTok[id] !== tok) return;
            self._observeResize(id, spec);
            return doEmbed();
        });
    },

    // Runs fn once the HTML parser has finished the page. A plot embedded by
    // an inline script while the page loads would otherwise be measured
    // before the parser reaches the rest of its container: the plots after
    // it in a grid or flex row do not exist yet, so an auto-fit grid or a
    // wrapping flex row hands the first plot the whole row. An htmx swap
    // inserts its whole fragment before running its scripts, so embeds
    // after load run fn at once.
    _afterParse: function(fn) {
        if (document.readyState !== 'loading') return fn();
        return new Promise(function(resolve, reject) {
            document.addEventListener('DOMContentLoaded', function() {
                try { resolve(fn()); } catch (err) { reject(err); }
            }, {once: true});
        });
    },

    // Set up resize observer for responsive re-embed: container-sized
    // specs, and _aov-marked ones (a remap can make them container-sized).
    // A plot whose container offered no width when it embedded re-sizes
    // as soon as it does (_measurableAgain); other width changes are
    // coalesced.
    _observeResize: function(id, spec) {
        var self = this;
        if (spec._aov || self._fitSized(spec)) {
            var el = document.getElementById(id);
            if (el && el.parentElement && window.ResizeObserver) {
                var timer = null;
                var lastWidth = el.parentElement.clientWidth;
                var ro = new ResizeObserver(function() {
                    var newWidth = el.parentElement ? el.parentElement.clientWidth : 0;
                    if (self._measurableAgain(id, el)) {
                        lastWidth = newWidth;
                        clearTimeout(timer);
                        self._resizeReembed(id);
                        return;
                    }
                    if (newWidth === lastWidth || newWidth < 50) return;
                    lastWidth = newWidth;
                    clearTimeout(timer);
                    timer = setTimeout(function() { self._resizeReembed(id); }, 200);
                });
                ro.observe(el.parentElement);
                // Clean up on element removal
                self._observers = self._observers || {};
                if (self._observers[id]) { self._observers[id].disconnect(); }
                self._observers[id] = ro;
            }
        }
    },

    // --- Keyboard access to a plot's scroll frame ---
    // A plot wider than its `.aov-plot-area` scrolls inside it (aov.css), and a
    // canvas holds nothing that takes the focus. So while the frame scrolls it
    // is a named tab stop itself, where the arrow keys scroll it. Without that,
    // Chrome reaches it as a stop with no name, and a browser that does not
    // make scrollers focusable does not reach it.
    // A plot that fits adds no stop. The frame outlives a re-embed, so the
    // reader's focus does too. A name needs a role, and
    // `figure` is what the frame holds. Not `group`: Pico lays `[role=group]`
    // out as an inline-flex button row. Not `region`: a landmark per plot, and
    // two untitled plots then fail axe's `landmark-unique`.
    _watchScrollFrame: function(id) {
        var self = this, el = self._els[id], area = el && el.closest('.aov-plot-area');
        if (!area) return;
        if (self._scrollFrames[id]) self._scrollFrames[id].disconnect();
        if (window.ResizeObserver) {
            // The frame for a changed container, the drawing for a view that
            // grew with its data (each embed draws a new one).
            var ro = self._scrollFrames[id] = new ResizeObserver(function() { self._syncScrollFrame(id); });
            ro.observe(area);
            var marks = el.querySelector('.marks');
            if (marks) ro.observe(marks);
        }
        self._syncScrollFrame(id);
    },
    _syncScrollFrame: function(id) {
        // A plot whose width follows data still loading from a URL fits its
        // frame until the data arrives, and dropping the tab stop then would
        // drop the reader's focus. The embed measures again when it settles.
        var self = this, el = this._els[id], area = el && el.closest('.aov-plot-area');
        if (!area || this._inFlight[id]) return;
        this._syncFrameStop(area, function() { return self._scrollFrameName(id, area); });
    },
    // A frame is a named tab stop while it scrolls, and no stop otherwise.
    _syncFrameStop: function(frame, name) {
        // Within 2 px is no scrolling to speak of (the tolerance of
        // _fitCorrection). The computed overflow keeps a frame on a page
        // without aov.css, which does not scroll, out of the tab order.
        var style = getComputedStyle(frame);
        var scrolls = function(overflow, over) { return (overflow === 'auto' || overflow === 'scroll') && over > 2; };
        if (scrolls(style.overflowX, frame.scrollWidth - frame.clientWidth) ||
            scrolls(style.overflowY, frame.scrollHeight - frame.clientHeight)) {
            frame.setAttribute('tabindex', '0');
            frame.setAttribute('role', 'figure');
            frame.setAttribute('aria-label', name());
        } else {
            frame.removeAttribute('tabindex');
            frame.removeAttribute('role');
            frame.removeAttribute('aria-label');
        }
    },
    // The title of the caption of the figure around `el` (as the HTML download
    // reads it).
    _captionTitle: function(el) {
        var figure = el.closest('figure');
        var caption = figure && figure.querySelector('figcaption .caption-header strong');
        return caption && caption.textContent.trim();
    },
    // "Scrollable plot", then what the reader knows the plot by: its caption's
    // title, or else the spec's own title.
    _scrollFrameName: function(id, area) {
        var word = this._texts(area)('scrollable');
        var title = this._captionTitle(area);
        if (!title) {
            title = this._cur[id] && this._cur[id].spec.title;
            if (title && title.text !== undefined) title = title.text;
            if (Array.isArray(title)) title = title.join(' ');
        }
        return title ? word + ': ' + title : word;
    },

    // A plot's container changed width: re-embed it at the new width when it is
    // container-sized. A fixed-width plot is left alone — a re-embed would only
    // replace its view and drop its legend selection — unless it is an
    // auto-wrapped facet whose row now fits a different number of panels.
    // Reads the current spec, so a remap can turn a plot responsive or back.
    // Returns whether it re-embedded.
    _resizeReembed: function(id) {
        var cur = this._cur[id], again = this._reembeds[id];
        if (!cur || !again) return false;
        if (!this._fitSized(cur.spec) &&
            !(this._autoColumns(cur.spec) && this._wrapChanged(id, cur.spec))) return false;
        again();
        return true;
    },

    // Whether an auto-wrapped facet with authored panel widths would now wrap
    // into a different number of columns than its last embed estimated. One
    // never sized (embedded while hidden) is due.
    _wrapChanged: function(id, spec) {
        var info = (this._computedWidths || {})[id];
        if (!info || info.kind !== 'wrap') return true;
        var el = document.getElementById(id);
        var budget = el ? this._sizingBudget(el, spec) : null;
        return budget !== null && this._wrapColumns(spec, budget, false) !== info.estimate;
    },

    // --- Host theme ---
    // The page's light/dark choice reaches a plot through the CSS `color`
    // its element inherits. Vega's canvas cannot resolve `currentColor`
    // (it paints black), so the runtime resolves the colour itself and
    // embeds with a Vega-Lite config that draws the chrome (axes, legends,
    // facet headers, titles, unencoded text) in it on a transparent
    // background, in the element's font. A page may also restyle plots
    // through `--aov-*` custom properties (_HOST_LOOK). Mark colours are
    // left alone, apart from the default categorical range a page may set
    // and the first default colour on a dark page (_darkCategory).
    // The spec's own `config` wins key by key (vega-lite merges the embed
    // config under it).
    // Page default: `window.AoV.theme` (`vega_head(theme=...)`); per plot:
    // `spec._aov.theme` (`config(theme=...)`). Values: 'host' or 'none'.
    _themeOn: function(spec) {
        var t = spec && spec._aov && spec._aov.theme;
        if (t === undefined || t === null) t = window.AoV && window.AoV.theme;
        return (t === undefined || t === null ? 'host' : t) === 'host';
    },
    _rgba: function(css) {
        var m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:\s*[,\/]\s*([\d.]+)(%?))?\s*\)$/.exec(css || '');
        if (!m) return null;
        var a = m[4] === undefined ? 1 : parseFloat(m[4]) / (m[5] ? 100 : 1);
        return [+m[1], +m[2], +m[3], a];
    },
    _withAlpha: function(c, a) {
        return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (+(c[3] * a).toFixed(3)) + ')';
    },
    // WCAG relative luminance of a colour from _rgba (its alpha is ignored).
    _luminance: function(c) {
        var lin = function(v) { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
        return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
    },
    // Light text (closer to white than to black by WCAG contrast): the plot
    // sits on a dark page.
    _lightText: function(c) {
        return this._luminance(c) > 0.179;
    },
    // Vega's default categorical palette (tableau10) and default mark colour
    // are made for white. On a dark page its first colour, #4c78a8, is a
    // greyish blue (OKLCH chroma .089) at nearly the hue of a blue-grey dark
    // background (3.9:1 on Pico's #13171f): lines read as navy and
    // translucent bands as grey. A dark page draws that slot as #4893e2, the
    // same hue lighter and more saturated (OKLCH .65/.14, 5.6:1 there, 3.9:1
    // on #333); the other nine colours read on dark pages as they are.
    _DARK_FIRST: '#4893e2',
    _darkCategory: function() {
        var cat = (typeof vega !== 'undefined' && vega.scheme && vega.scheme('tableau10')) || null;
        if (!cat) return null;
        cat = cat.slice();
        cat[0] = this._DARK_FIRST;
        return cat;
    },
    // The theme vega-tooltip shows the tooltips of a plot in `el` with, so
    // that a tooltip reads as one surface in the page's own colours, as the
    // plot's chrome does: the background behind the plot and its text colour.
    // The surface is opaque: under clear cells, vega-tooltip's .95 lets the
    // page's text show through the tooltip's. The border reproduces its light
    // look for black text on white (#d9d9d9). Keys are the text colour at .75,
    // not the .5 that vega-tooltip's #808080 is on white: on Pico's text colours
    // .65 and below fall under WCAG AA contrast (3.95:1 on its light page).
    // All plots share one tooltip element, and vega-tooltip sets
    // `<theme>-theme` on it while a plot's tooltip shows, so every distinct
    // look is a theme of its own: one rule, written once. The rule ranks with
    // vega-tooltip's defaults and comes right after them, so a page's own
    // `#vg-tooltip-element` rules still win.
    _tooltipTheme: function(el) {
        var cs = el ? getComputedStyle(el) : null;
        var text = (cs && this._rgba(cs.color)) || [0, 0, 0, 1];
        var back = (el && this._rgba(this.hostBackground(el))) || [255, 255, 255, 1];
        // Text that cannot be read on that background is not read on it: a
        // background image or gradient, a translucent card (hostBackground
        // takes its colour as opaque) or a dark `color-scheme` page that
        // paints no background of its own. Pick the surface for the text.
        var lt = this._luminance(text), lb = this._luminance(back);
        if ((Math.max(lt, lb) + 0.05) / (Math.min(lt, lb) + 0.05) < 3) {
            back = this._lightText(text) ? [32, 32, 32, 1] : [255, 255, 255, 1];
        }
        var surface = this._withAlpha(back, 1), color = this._withAlpha(text, 1);
        var border = this._withAlpha(text, 0.15), key = this._withAlpha(text, 0.75);
        var look = [surface, color, border, key].join('|');
        var theme = this._tipThemes[look];
        if (theme) return theme;
        theme = this._tipThemes[look] = 'aov' + Object.keys(this._tipThemes).length;
        if (!this._tipStyle) {
            this._tipStyle = document.createElement('style');
            var head = document.head, defaults = document.getElementById('vega-tooltip-style');
            head.insertBefore(this._tipStyle, defaults ? defaults.nextSibling : head.firstChild);
        }
        var sel = '#vg-tooltip-element:where(.' + theme + '-theme)';
        this._tipStyle.appendChild(document.createTextNode(
            sel + '{background-color:' + surface + ';border-color:' + border + ';color:' + color + '}' +
            sel + ' table tr td.key{color:' + key + '}'));
        return theme;
    },
    // The Vega-Lite config for a plot element's resolved text colour, font
    // and look (_hostLook). Alphas reproduce Vega's defaults for black text
    // on white (domain and ticks #888, grid and view frame #ddd), so a
    // light page renders as before apart from the transparent background.
    // On a dark page (light text) the default categorical range and the
    // default mark colour take _darkCategory; a page's `--aov-category`
    // replaces that range as on a light page.
    hostThemeConfig: function(el) {
        var cs = el ? getComputedStyle(el) : null;
        var c = (cs && this._rgba(cs.color)) || [0, 0, 0, 1];
        var text = this._withAlpha(c, 1), soft = this._withAlpha(c, 0.467), faint = this._withAlpha(c, 0.133);
        var cfg = {
            background: 'transparent',
            title: {color: text, subtitleColor: soft},
            axis: {labelColor: text, titleColor: text, domainColor: soft, tickColor: soft, gridColor: faint},
            legend: {labelColor: text, titleColor: text},
            header: {labelColor: text, titleColor: text},
            view: {stroke: faint},
            text: {color: text},
            style: {'aov-ink': {color: text}, 'aov-ink-stroke': {stroke: text}, 'aov-ink-fill': {fill: text}}
        };
        var dark = this._lightText(c) && this._darkCategory();
        if (dark) {
            cfg.range = {category: dark};
            cfg.mark = {color: dark[0]};
        }
        if (cs) this._hostLook(cfg, el, cs);
        return cfg;
    },
    // A page's plot look: CSS custom properties set on the plot element or
    // any ancestor (`html[data-look=x] { --aov-grid-color: #eee }`), read
    // from its computed style, so `var(...)` tokens, `currentColor` and
    // `light-dark()` resolve where the plot sits. Each row: property,
    // config group, config key, and the group's settings that `none`
    // applies (the part leaves the plot).
    // `--aov-category` (a comma-separated colour list) is the default
    // categorical range: explicit scale ranges and mark colours keep theirs.
    // LOCKSTEP: the property table in `vega_head`'s docstring
    // (src/js_runtime.jl) and test/host_look.js.
    _HOST_LOOK: [
        ['--aov-axis-label-color', 'axis', 'labelColor'],
        ['--aov-axis-title-color', 'axis', 'titleColor'],
        ['--aov-grid-color', 'axis', 'gridColor', {grid: false}],
        ['--aov-tick-color', 'axis', 'tickColor', {ticks: false}],
        ['--aov-domain-color', 'axis', 'domainColor', {domain: false}],
        ['--aov-view-color', 'view', 'stroke', {stroke: null}],
        ['--aov-legend-label-color', 'legend', 'labelColor'],
        ['--aov-legend-title-color', 'legend', 'titleColor'],
        ['--aov-header-label-color', 'header', 'labelColor'],
        ['--aov-header-title-color', 'header', 'titleColor']
    ],
    _hostLook: function(cfg, el, cs) {
        var probe = null;
        // A CSS value as the browser computes it on a hidden child of `el`
        // (null when it is no valid `prop`).
        var computed = function(prop, value) {
            if (!probe) {
                probe = document.createElement('span');
                probe.style.display = 'none';
                el.appendChild(probe);
            }
            probe.style[prop] = '';
            probe.style[prop] = value;
            return probe.style[prop] ? getComputedStyle(probe)[prop] : null;
        };
        var invalid = function(name, value, what) {
            console.warn('AoV: ' + name + ' is not ' + what + ': ' + value);
        };
        try {
            var font = this._hostFont(cs, computed, invalid);
            if (font) cfg.font = font;
            this._HOST_LOOK.forEach(function(p) {
                var v = cs.getPropertyValue(p[0]).trim();
                if (!v) return;
                if (p[3] && v.toLowerCase() === 'none') { Object.assign(cfg[p[1]], p[3]); return; }
                var color = computed('color', v);
                if (color) cfg[p[1]][p[2]] = color; else invalid(p[0], v, 'a CSS colour');
            });
            var cat = cs.getPropertyValue('--aov-category').trim();
            if (cat) {
                var colors = this._splitTopLevel(cat).map(function(v) {
                    var color = computed('color', v);
                    if (!color) invalid('--aov-category', v, 'a CSS colour');
                    return color;
                });
                // A palette with a hole would shift every later category's colour.
                if (colors.length && colors.every(Boolean)) cfg.range = {category: colors};
            }
        } finally {
            if (probe) probe.remove();
        }
    },
    // The plot font: `--aov-font`, else the element's own font-family when
    // the page set one. A page left on the browser's default font (Chrome:
    // "Times New Roman") keeps Vega's sans-serif.
    _hostFont: function(cs, computed, invalid) {
        var own = cs.getPropertyValue('--aov-font').trim();
        if (own) {
            var font = computed('fontFamily', own);
            if (!font) invalid('--aov-font', own, 'a CSS font-family');
            return font;
        }
        if (this._uaFont === undefined) this._uaFont = computed('fontFamily', 'initial');
        return cs.fontFamily && cs.fontFamily !== this._uaFont ? cs.fontFamily : null;
    },
    // `a, rgb(1, 2, 3), b` → ['a', 'rgb(1, 2, 3)', 'b']: commas inside
    // parentheses belong to their colour.
    _splitTopLevel: function(value) {
        var out = [], depth = 0, start = 0;
        for (var i = 0; i < value.length; i++) {
            var ch = value.charAt(i);
            if (ch === '(') depth++;
            else if (ch === ')') depth--;
            else if (ch === ',' && depth === 0) { out.push(value.slice(start, i)); start = i + 1; }
        }
        out.push(value.slice(start));
        return out.map(function(s) { return s.trim(); }).filter(Boolean);
    },
    // A promise that settles once the faces `font` draws plot text with
    // (regular and bold) are loaded, or null when none is pending. Canvas
    // text drawn before a web font loads stays in the fallback font, and
    // its fallback widths size the layout, so the embed waits for it.
    _fontsPending: function(font) {
        var fonts = typeof document !== 'undefined' && document.fonts;
        if (!font || !fonts || !fonts.check) return null;
        var faces = ['10px ' + font, 'bold 10px ' + font];
        try {
            if (faces.every(function(f) { return fonts.check(f); })) return null;
        } catch (err) {
            console.warn('AoV: plot font ' + font + ' is not a CSS font-family:', err);
            return null;
        }
        return Promise.all(faces.map(function(f) { return fonts.load(f); })).catch(function(err) {
            console.warn('AoV: plot font ' + font + ' did not load; drawing with the fallback:', err);
        });
    },
    // The first opaque background behind `el` (exports paint it).
    hostBackground: function(el) {
        for (var n = el; n && n.nodeType === 1; n = n.parentElement) {
            var c = this._rgba(getComputedStyle(n).backgroundColor);
            if (c && c[3] > 0) return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
        }
        return 'white';
    },
    _deepMerge: function(base, over) {
        var out = Object.assign({}, base);
        Object.keys(over || {}).forEach(function(k) {
            var b = out[k], o = over[k];
            out[k] = (b && o && typeof b === 'object' && typeof o === 'object' &&
                !Array.isArray(b) && !Array.isArray(o)) ? this._deepMerge(b, o) : o;
        }, this);
        return out;
    },
    // AoV's theme inks in `spec` (a copy being embedded) take the host
    // theme's colour unless the spec set a colour other than AoV's `#333`:
    // `aov-ink` (`color=:ink`, dot-interval rules), a style Vega-Lite
    // resolves, and `aov-ink-stroke` / `aov-ink-fill` (`stroke=:ink`,
    // `fill=:ink`), whose colour the server also wrote onto every mark that
    // carries them. LOCKSTEP: `_AOV_INK_STYLES` (src/to_vegalite.jl).
    _hostInk: function(spec, theme) {
        var styles = spec.config && spec.config.style;
        if (!styles) return;
        var paint = [];
        Object.keys(theme.style).forEach(function(name) {
            var prop = Object.keys(theme.style[name])[0], own = styles[name];
            if (!own || own[prop] !== '#333') return;
            own[prop] = theme.style[name][prop];
            if (prop !== 'color') paint.push([name, prop, own[prop]]);
        });
        if (!paint.length) return;
        (function walk(node) {
            if (Array.isArray(node)) { node.forEach(walk); return; }
            if (!node || typeof node !== 'object') return;
            var style = node.style;
            paint.forEach(function(p) {
                var styled = style === p[0] || (Array.isArray(style) && style.indexOf(p[0]) !== -1);
                if (styled && node[p[1]] === '#333') node[p[1]] = p[2];
            });
            Object.keys(node).forEach(function(k) {
                if (k !== 'data' && k !== 'datasets' && k !== 'config') walk(node[k]);
            });
        })(spec);
    },
    // `opts` for one embed of `spec` into plot `id`, with the host theme
    // applied. Records the theme so refreshTheme can tell when it changed.
    _themedOpts: function(id, spec, opts) {
        if (!this._themeOn(spec)) { delete this._themeKeys[id]; return opts; }
        var el = this._els[id] || document.getElementById(id);
        var cfg = this.hostThemeConfig(el), tip = this._tooltipTheme(el);
        this._themeKeys[id] = this._themeKey(cfg, tip);
        this._hostInk(spec, cfg);
        var themed = Object.assign({}, opts, {config: this._deepMerge(cfg, opts.config)});
        if (opts.tooltip === undefined) themed.tooltip = {theme: tip};
        return themed;
    },
    // What a themed drawing depends on: its config and its tooltip's look.
    _themeKey: function(cfg, tip) {
        return JSON.stringify(Object.assign({tooltip: tip}, cfg));
    },
    // Re-embed every themed plot whose host colours, font or look changed. Runs on
    // prefers-color-scheme changes and on class / data-theme / style
    // changes of <html> and <body>; call it after switching a theme by
    // other means (e.g. a class on a wrapper element).
    refreshTheme: function() {
        var self = this;
        Object.keys(self._themeKeys).forEach(function(id) {
            var el = self._els[id], again = self._reembeds[id];
            if (!el || !el.isConnected || !again) return;
            if (self._themeKey(self.hostThemeConfig(el), self._tooltipTheme(el)) !== self._themeKeys[id]) again();
        });
        self._refreshCellThemes();
    },
    _watchTheme: function() {
        if (this._themeWatch || typeof document === 'undefined') return;
        var self = this, frame = null;
        var check = function() {
            if (frame !== null) return;
            var raf = window.requestAnimationFrame || function(f) { return setTimeout(f, 16); };
            frame = raf(function() { frame = null; self.refreshTheme(); });
            // A CSS colour transition settles after the first frame.
            setTimeout(function() { self.refreshTheme(); }, 400);
        };
        this._themeWatch = check;
        if (window.matchMedia) {
            var mq = window.matchMedia('(prefers-color-scheme: dark)');
            if (mq.addEventListener) mq.addEventListener('change', check); else if (mq.addListener) mq.addListener(check);
        }
        if (typeof MutationObserver !== 'undefined') {
            var mo = new MutationObserver(check);
            [document.documentElement, document.body].forEach(function(n) {
                if (n) mo.observe(n, {attributes: true, attributeFilter: ['class', 'data-theme', 'style']});
            });
        }
    },

    // Vega-Lite compiles inside vegaEmbed with no per-embed logger, so its
    // warnings carry no plot id. One permanent wrapper tags them while
    // exactly one embed is in flight (with several in flight the source is
    // ambiguous). Per-embed save/restore of console.warn interleaves across
    // concurrent embeds and leaves a growing chain of wrappers installed,
    // each keeping its plot reachable after dispose().
    _inFlight: {},
    _tagConsole: function() {
        if (this._console) return;
        var self = this, orig = this._console = {warn: console.warn, error: console.error};
        ['warn', 'error'].forEach(function(level) {
            console[level] = function() {
                var a = Array.prototype.slice.call(arguments), ids = Object.keys(self._inFlight);
                if (ids.length === 1) a[0] = '[' + ids[0] + '] ' + a[0];
                return orig[level].apply(console, a);
            };
        });
    },

    // Finalize a plot's current view through vega-embed's own finalize:
    // View.finalize() alone leaves the actions menu's document click
    // listener, which keeps the whole replaced view reachable.
    _finalizeView: function(id) {
        var r = this._results[id], v = this.views[id];
        if (r) r.finalize(); else if (v) v.finalize();
        delete this._results[id];
        delete this.views[id];
    },

    // Tear a plot down: finalize its view (removing the window/document
    // listeners that otherwise keep its data, scenegraph and canvas alive)
    // and drop every piece of per-plot state. A plot whose element stays in
    // the page keeps its last frame as a static copy (_releaseDrawing); an
    // embed still in flight is finalized, and its drawing released, when it
    // resolves.
    dispose: function(id) {
        var self = this, el = self._els[id];
        self._touchReset(id);
        self._hoverEnd(id, null);
        self._frameStop(id);
        self._finalizeView(id);
        if (self._observers && self._observers[id]) self._observers[id].disconnect();
        if (self._scrollFrames[id]) self._scrollFrames[id].disconnect();
        self._perPlotState.forEach(function(k) { if (self[k]) delete self[k][id]; });
        self._releaseDrawing(id, el);
    },

    // A finalized view stays reachable from the drawing it leaves in the page:
    // Vega 5.33.1's View.finalize removes none of the listeners its canvas
    // handler put on the canvas (canvas → listener → handler → view), nor
    // the svg handler's listeners and the scenegraph item every svg node
    // carries as `__data__`; vega-embed's finalize leaves the actions menu,
    // whose export links close over the view. So a plot disposed while its
    // element stays in the page swaps its drawing for a static copy: a canvas
    // with the same attributes and pixels, an svg clone (cloneNode copies
    // neither listeners nor `__data__`), and no actions menu. A detached
    // element is collectable as it is and is left alone. Only vega-embed's
    // public DOM (`.marks`, `.vega-actions`) is read, no private Vega field.
    _releaseDrawing: function(id, el) {
        if (!el || !el.isConnected) return;
        var self = this;
        el.querySelectorAll('canvas.marks').forEach(function(canvas) {
            try {
                var copy = document.createElement('canvas');
                Array.prototype.forEach.call(canvas.attributes, function(a) { copy.setAttribute(a.name, a.value); });
                if (canvas.width && canvas.height) {
                    var ctx = copy.getContext('2d');
                    if (!ctx) throw new Error('no 2d context for a ' + canvas.width + 'x' + canvas.height + ' canvas');
                    ctx.drawImage(canvas, 0, 0);
                }
                canvas.replaceWith(copy);
                // The replaced canvas's bitmap goes now, not when it is
                // collected. Without this, headless Chrome kept a detached
                // canvas, and so its view, reachable for a while in about 1
                // run in 10 (test/view_retention.js).
                canvas.width = canvas.height = 0;
            } catch (err) {
                // The last frame stays painted, and the canvas keeps its view.
                (self._console || console).error.call(console, '[' + id + '] AoV.dispose could not copy the drawing; ' +
                    'the plot keeps its Vega view while the drawing stays in the page:', err);
            }
        });
        el.querySelectorAll('svg.marks').forEach(function(svg) { svg.replaceWith(svg.cloneNode(true)); });
        el.querySelectorAll('.vega-actions').forEach(function(menu) {
            var details = menu.closest('details');
            (details && el.contains(details) ? details : menu).remove();
        });
    },

    // Dispose every plot whose element is `root` or inside it. Walks the
    // registry rather than the DOM, so it also works on a detached subtree.
    disposeWithin: function(root) {
        var self = this;
        if (!root) return;
        Object.keys(self._els).forEach(function(id) {
            var el = self._els[id];
            if (el && (el === root || root.contains(el))) self.dispose(id);
        });
    },

    whenReady: function(id, fn) {
        var view = this.views[id];
        if (view) { fn(view); return; }
        this._pending[id] = this._pending[id] || [];
        this._pending[id].push(fn);
    },

    updateData: function(id, data, name) {
        name = name || 'source_0';
        data = this._rowsFromColumns(data);
        var self = this;
        this.whenReady(id, function(view) {
            if (self._refuseFramed('updateData', id) ||
                    self._refuseRawRows('updateData', id, view, name, data)) return;
            self._applyRows(id, view, name, data,
                vega.changeset().remove(function() { return true; }).insert(data));
        });
    },

    appendData: function(id, data, name, maxRows) {
        name = name || 'source_0';
        data = this._rowsFromColumns(data);
        var self = this;
        this.whenReady(id, function(view) {
            if (self._refuseFramed('appendData', id) ||
                    self._refuseRawRows('appendData', id, view, name, data)) return;
            var rows = self._currentRows(id, name).concat(data);
            var trimmed = maxRows && rows.length > maxRows;
            if (trimmed) rows = rows.slice(rows.length - maxRows);
            self._applyRows(id, view, name, rows, trimmed ?
                vega.changeset().remove(function() { return true; }).insert(rows) :
                vega.changeset().insert(data));
        });
    },

    // Rows sent column by column (`_wire_columns` in to_node.jl):
    // {n, columns: {name: column}}, where a column is an array of values, a
    // scalar shared by every row, {runs, lengths}, or a numeric sequence
    // {start, step, scale?, den?, period?} whose row i holds
    // ((start + j*step)*scale)/den with j = i % period. Returns the row
    // objects the row-wise JSON would have parsed to; a row array passes
    // through unchanged. The server picks a sequence only when exactly
    // these double operations reproduce every value.
    _isColumns: function(data) {
        return !!data && typeof data === 'object' && !Array.isArray(data) &&
            typeof data.n === 'number' && !!data.columns && typeof data.columns === 'object';
    },
    _rowsFromColumns: function(data) {
        if (!this._isColumns(data)) return data;
        var n = data.n, names = Object.keys(data.columns), self = this;
        var cols = names.map(function(k) { return self._columnValues(k, data.columns[k], n); });
        var rows = new Array(n);
        for (var i = 0; i < n; i++) {
            var row = {};
            for (var j = 0; j < names.length; j++) row[names[j]] = cols[j][i];
            rows[i] = row;
        }
        return rows;
    },
    _columnValues: function(name, c, n) {
        if (Array.isArray(c)) return c;
        var out = new Array(n), i;
        if (c === null || typeof c !== 'object') {
            for (i = 0; i < n; i++) out[i] = c;
        } else if (Array.isArray(c.runs)) {
            var k = 0;
            c.runs.forEach(function(v, r) { for (var m = 0; m < c.lengths[r]; m++) out[k++] = v; });
        } else if (typeof c.start === 'number' && typeof c.step === 'number') {
            var p = c.period || n;
            for (i = 0; i < n; i++) {
                var v = c.start + (i % p) * c.step;
                if (c.scale !== undefined) v *= c.scale;
                if (c.den !== undefined) v /= c.den;
                out[i] = v;
            }
        } else {
            throw new Error('AoV: column "' + name + '" has an unknown encoding with keys ' +
                Object.keys(c).join(', '));
        }
        return out;
    },
    // A spec's inline datasets (`data.values`, `datasets` entries) sent
    // column by column, expanded in place to rows.
    _expandSpecData: function(spec) {
        var self = this;
        (function walk(v) {
            if (Array.isArray(v)) { v.forEach(walk); return; }
            if (!v || typeof v !== 'object') return;
            Object.keys(v).forEach(function(k) {
                var x = v[k];
                if (k === 'data' && x && typeof x === 'object' && !Array.isArray(x)) {
                    if (self._isColumns(x.values)) x.values = self._rowsFromColumns(x.values);
                    else if (!Array.isArray(x.values)) walk(x);
                } else if (k === 'datasets' && x && typeof x === 'object') {
                    Object.keys(x).forEach(function(name) {
                        if (self._isColumns(x[name])) x[name] = self._rowsFromColumns(x[name]);
                    });
                } else {
                    walk(x);
                }
            });
        })(spec);
        return spec;
    },

    // Replace the rows of the groups `data` carries (`replace_data`): every
    // row whose `key` field values equal those of a row of `data` is
    // removed and `data` inserted, in one changeset; the rows of every
    // other group stay in the view untouched.
    _keyToken: function(row, fields) {
        return JSON.stringify(fields.map(function(f) { return row[f]; }));
    },
    _hasKeyFields: function(row, fields) {
        return fields.every(function(f) { return Object.prototype.hasOwnProperty.call(row, f); });
    },
    replaceData: function(id, data, key, name) {
        name = name || 'source_0';
        data = this._rowsFromColumns(data);
        var self = this, fields = [].concat(key);
        this.whenReady(id, function(view) {
            if (self._refuseFramed('replaceData', id) ||
                    self._refuseRawRows('replaceData', id, view, name, data)) return;
            var keys = new Set(data.map(function(r) { return self._keyToken(r, fields); }));
            var replaced = function(r) {
                return self._hasKeyFields(r, fields) && keys.has(self._keyToken(r, fields));
            };
            var rows = self._currentRows(id, name).filter(function(r) { return !replaced(r); });
            self._applyRows(id, view, name, rows.concat(data),
                vega.changeset().remove(replaced).insert(data));
        });
    },

    // Remove one explicitly named group without shipping replacement rows.
    // A missing key field is an error instead of a silent no-op (which would
    // be easy to mistake for a successful removal from a lowered dataset).
    removeData: function(id, values, key, name) {
        name = name || 'source_0';
        var self = this, fields = [].concat(key), token = JSON.stringify(values);
        this.whenReady(id, function(view) {
            if (self._refuseFramed('removeData', id)) return;
            var current = self._currentRows(id, name);
            if (current.length && !current.some(function(r) { return self._hasKeyFields(r, fields); })) {
                console.error('AoV.removeData: dataset "' + name + '" of plot "' + id +
                    '" lacks key field(s) ' + fields.join(', ') + '. Data left unchanged.');
                return;
            }
            var removed = function(r) {
                return self._hasKeyFields(r, fields) && self._keyToken(r, fields) === token;
            };
            if (!current.some(removed)) return;
            self._applyRows(id, view, name, current.filter(function(r) { return !removed(r); }),
                vega.changeset().remove(removed));
        });
    },

    // A dataset's rows as the plot shows them: those a data operation set,
    // else the embedded spec's own.
    _currentRows: function(id, name) {
        return (this._liveRows[id] || {})[name] || (this._specRows[id] || {})[name] || [];
    },

    // Record `rows` as the dataset's rows — re-embeds compile with them —
    // and apply `changeset`, the same change, to the live view. A coloured
    // ribbon whose rows bring a new colour group re-embeds instead, with
    // that group's layers (_regroupRibbons).
    _applyRows: function(id, view, name, rows, changeset) {
        var live = this._liveRows[id] = this._liveRows[id] || {};
        live[name] = rows;
        if (!this._regroupRibbons(id, name)) {
            this._runRelayout(view.change(name, changeset));
            this._maybeRestoreLegends(id);
        }
        this._dataChanged(id);
    },

    // A coloured lineribbon/ribbon draws each colour group as its own
    // layers — bands, then line, tagged `_lr_group` — so the groups paint
    // in order (`_ribbon_to_vl`); an empty first render has the
    // `_lr_proto` template layers instead. Rows that bring a group with no
    // layers would not be drawn, so re-embed with layers for the groups of
    // the rows. Returns whether it did. Such layers read the plot's
    // top-level rows, the `source_0` dataset.
    _regroupRibbons: function(id, name) {
        var cur = this._cur[id];
        if (name !== 'source_0' || !cur) return false;
        var rows = this._currentRows(id, name);
        var spec = this._withRibbonGroups(cur.spec, rows);
        if (!spec) { this._warnUngrownRibbons(id, cur.spec, rows); return false; }
        // A re-facet's re-embed keeps the true original (remapEncoding).
        var orig = this._origSpecs[id], remapped = !!this._mappings[id];
        this._embed(id, spec, this._embedOpts[id], true);
        if (remapped) this._origSpecs[id] = orig;
        return true;
    },
    // A ribbon layered with other layers (`lineribbon(...) + visual(...)`)
    // keeps its group layers each with its own copy of the rows, which a
    // data change cannot extend: say so when the rows bring a group it lacks.
    _warnUngrownRibbons: function(id, spec, rows) {
        var self = this, host = spec && spec.spec && Array.isArray(spec.spec.layer) ? spec.spec : spec;
        var have = {}, missing = [];
        ((host && host.layer) || []).forEach(function(l) {
            if (!l || !l._lr_layer || !l.data || l._lr_group === undefined ||
                !l.encoding || !l.encoding.color || typeof l.encoding.color.field !== 'string') return;
            var f = l.encoding.color.field;
            (have[f] = have[f] || {})[JSON.stringify(l._lr_group)] = true;
        });
        Object.keys(have).forEach(function(f) {
            self._rowGroups(rows, f).forEach(function(g) {
                if (!have[f][JSON.stringify(g)]) missing.push(g);
            });
        });
        if (missing.length) console.warn('AoV: the rows for plot "' + id + '" bring colour group(s) ' +
            missing.join(', ') + ' that its layered lineribbon/ribbon has no layers for, so they are ' +
            'not drawn there. Send update_spec(id, spec) when a group first appears in a ribbon ' +
            'layered with other layers.');
    },
    _isRibbonGroupLayer: function(l) {
        return !!(l && l._lr_layer && !l.data && (l._lr_group !== undefined || l._lr_proto) &&
            l.encoding && l.encoding.color && typeof l.encoding.color.field === 'string');
    },
    // The runs of consecutive group layers of one colour field in `layers`.
    _ribbonRuns: function(layers) {
        var runs = [], i = 0;
        while (i < layers.length) {
            if (!this._isRibbonGroupLayer(layers[i])) { i++; continue; }
            var field = layers[i].encoding.color.field, j = i;
            while (j < layers.length && this._isRibbonGroupLayer(layers[j]) &&
                layers[j].encoding.color.field === field) j++;
            runs.push({start: i, end: j, field: field});
            i = j;
        }
        return runs;
    },
    // The colour groups of `rows` in `field`, in the order the server
    // emits group layers (sorted).
    _rowGroups: function(rows, field) {
        var seen = {}, groups = [];
        rows.forEach(function(r) {
            var v = r && r[field];
            if (v === undefined || v === null) return;
            var k = JSON.stringify(v);
            if (!seen[k]) { seen[k] = true; groups.push(v); }
        });
        return groups.sort(function(a, b) {
            if (typeof a === 'number' && typeof b === 'number') return a - b;
            a = String(a); b = String(b);
            return a < b ? -1 : a > b ? 1 : 0;
        });
    },
    // `spec` (a copy) with each ribbon's group layers rebuilt for the
    // groups of `rows`, or null when every group of `rows` already has
    // its layers. Existing groups keep their layers; a new group's layers
    // copy another group's (or the prototype layers) with its own filter.
    _withRibbonGroups: function(spec, rows) {
        var self = this;
        if (!spec || !spec.data || !Array.isArray(spec.data.values) || !rows.length) return null;
        var lacking = function(s) {
            var host = s.spec && Array.isArray(s.spec.layer) ? s.spec : s;
            var layers = Array.isArray(host.layer) ? host.layer : [];
            return self._ribbonRuns(layers).some(function(run) {
                var have = {};
                layers.slice(run.start, run.end).forEach(function(l) {
                    if (l._lr_group !== undefined) have[JSON.stringify(l._lr_group)] = true;
                });
                return self._rowGroups(rows, run.field).some(function(g) { return !have[JSON.stringify(g)]; });
            });
        };
        if (!lacking(spec)) return null;
        var out = JSON.parse(JSON.stringify(spec));
        this._stripLegendSelections(out);
        var host = out.spec && Array.isArray(out.spec.layer) ? out.spec : out;
        var layers = host.layer, result = [], at = 0;
        this._ribbonRuns(layers).forEach(function(run) {
            result = result.concat(layers.slice(at, run.start));
            at = run.end;
            var runLayers = layers.slice(run.start, run.end), byGroup = {}, proto = [];
            runLayers.forEach(function(l) {
                if (l._lr_proto) { proto.push(l); return; }
                var k = JSON.stringify(l._lr_group);
                (byGroup[k] = byGroup[k] || []).push(l);
            });
            if (!proto.length) proto = byGroup[JSON.stringify(runLayers[0]._lr_group)];
            var prefix = 'datum[' + JSON.stringify(run.field) + '] === ';
            proto = proto.map(function(l) {
                var c = JSON.parse(JSON.stringify(l));
                delete c.params; delete c._lr_group; delete c._lr_proto;
                c.transform = (c.transform || []).filter(function(t) {
                    return !(t && typeof t.filter === 'string' && t.filter.indexOf(prefix) === 0);
                });
                return c;
            });
            var next = [];
            self._rowGroups(rows, run.field).forEach(function(g) {
                next = next.concat(byGroup[JSON.stringify(g)] ||
                    self._ribbonGroupLayers(proto, run.field, [g]));
            });
            // Parameters (the zoom/pan binding) of layers that are dropped —
            // the prototype layers, a group the rows no longer carry — move
            // to the run's first layer.
            runLayers.forEach(function(l) {
                if (!l.params || next.indexOf(l) !== -1) return;
                next[0].params = (next[0].params || []).concat(l.params);
            });
            result = result.concat(next);
        });
        host.layer = result.concat(layers.slice(at));
        return out;
    },
    // One colour group's layers (`proto`: bands, then line) for each of
    // `groups`, each copy filtered to its group and tagged `_lr_group`.
    _ribbonGroupLayers: function(proto, field, groups) {
        var out = [];
        groups.forEach(function(g) {
            var expr = 'datum[' + JSON.stringify(field) + '] === ' + JSON.stringify(g);
            proto.forEach(function(tl) {
                var gl = JSON.parse(JSON.stringify(tl));
                gl.transform = (gl.transform || []).concat([{filter: expr}]);
                gl._lr_layer = true;
                gl._lr_group = g;
                out.push(gl);
            });
        });
        return out;
    },

    // Raw rows can only replace raw rows. A dataset AoV lowered server-side
    // (an interval/ribbon summary, a merged multi-layer dataset with its
    // `__src` filters, picker combo columns) carries fields raw rows lack:
    // inserting them would place or select no mark and blank the figure
    // without an error. Refuse, loudly, when the new rows lack a field
    // that places or selects marks and that the current rows carry.
    _refuseRawRows: function(fn, id, view, name, rows) {
        var current;
        try { current = view.data(name); } catch (e) { return false; }
        if (!current || !current.length || !rows || !rows.length) return false;
        var have = this._rowKeys(rows), had = this._rowKeys(current);
        var read = this._placementFields((this._cur[id] || {}).spec);
        var missing = Object.keys(had).filter(function(k) { return !have[k] && read[k]; });
        if (!missing.length) return false;
        console.error('AoV.' + fn + ': the rows for plot "' + id + '" lack ' + missing.join(', ') +
            ', which place or select its marks in dataset "' + name + '". That dataset was lowered ' +
            'by AoV (an interval/ribbon summary, a merged multi-layer dataset or picker combo ' +
            'columns), so raw rows cannot replace it: send update_spec(id, spec) instead (with ' +
            'auto_remap=... for a picker plot), which re-lowers the new data. Data left unchanged.');
        return true;
    },
    _rowKeys: function(rows) {
        var keys = {};
        rows.forEach(function(r) {
            if (r && typeof r === 'object') Object.keys(r).forEach(function(k) { keys[k] = true; });
        });
        return keys;
    },
    // Fields that place or select a Vega-Lite spec's marks: positional and
    // facet channel fields (encodings and facet operators) and the fields
    // a `filter` transform tests (`datum.x` / `datum["x"]` in an
    // expression, or a predicate's `field`). A row lacking one is not
    // drawn, or lands in a null panel; a row lacking only e.g. its colour
    // field — or a field a colour/opacity condition tests — still draws.
    _PLACEMENT_CHANNELS: ['x', 'y', 'x2', 'y2', 'theta', 'theta2', 'radius', 'radius2',
        'latitude', 'longitude', 'latitude2', 'longitude2', 'row', 'column', 'facet'],
    _placementFields: function(spec) {
        var out = {}, channels = this._PLACEMENT_CHANNELS;
        function take(def) {
            (Array.isArray(def) ? def : [def]).forEach(function(d) {
                if (d && typeof d === 'object' && typeof d.field === 'string') out[d.field] = true;
            });
        }
        (function walk(v, key, inFilter) {
            if (typeof v === 'string') {
                if (!inFilter) return;
                var re = /datum(?:\.([A-Za-z_$][\w$]*)|\[\s*(['"])(.*?)\2\s*\])/g, m;
                while ((m = re.exec(v))) out[m[1] || m[3]] = true;
            } else if (Array.isArray(v)) {
                v.forEach(function(x) { walk(x, key, inFilter); });
            } else if (v && typeof v === 'object') {
                if (key === 'encoding' || key === 'facet') {
                    channels.forEach(function(ch) { if (v[ch]) take(v[ch]); });
                    if (key === 'facet') take(v);
                }
                if (inFilter) take(v);
                Object.keys(v).forEach(function(k) {
                    if (k === 'values' && Array.isArray(v[k])) return;  // inline rows
                    walk(v[k], k, inFilter || k === 'filter');
                });
            }
        })(spec);
        return out;
    },

    // Refresh a plot from a newly lowered spec (`update_spec`), keeping the
    // reader's state. The picker's current channel assignment is re-applied
    // to the new spec. When the result differs from the embedded spec only
    // in inline data, the view's datasets are swapped in place, so zoom/pan,
    // legend selection and the canvas survive; any structural change
    // (panels, layers, encodings, size) re-embeds in place.
    updateSpec: function(id, spec, opts) {
        var self = this, el = document.getElementById(id);
        // Not embedded into this element yet (a first render, or a fresh
        // element from an htmx swap): an ordinary embed.
        if (el && self._els[id] !== el) return self.embed(id, spec, opts);
        self.whenReady(id, function(view) {
            var now = document.getElementById(id);
            // Sent while no element had this id: apply to the plot that
            // has since embedded under it (as appendData/updateData do).
            if (!el) { if (now) self.updateSpec(id, spec, opts); return; }
            // Removed, or replaced by a fresh render, meanwhile.
            if (now !== el || self._els[id] !== el) return;
            // A ribbon keeps the drawing the reader chose.
            var drawn = self._takeRibbonViews(id, spec, true);
            var pristine = self._expandSpecData(JSON.parse(JSON.stringify(drawn)));
            self._refreshLegendSelections(pristine);
            self._broadcastCrossSource(pristine);
            var shown = pristine;
            if (self._mappings[id]) {
                // The picker builds combo columns from the stored original.
                self._origSpecs[id] = pristine;
                var mapping = self._pickers[id] ? self._pickers[id]() : self._mappings[id];
                shown = (mapping && self._remappedSpec(id, mapping)) || pristine;
            }
            var prepared = JSON.parse(JSON.stringify(shown));
            self._refreshLegendSelections(prepared);
            self._broadcastCrossSource(prepared);
            var cur = self._cur[id];
            // A framed plot's fixed domains come from all its rows: re-embed.
            if (cur && !self._droppedLegends[id] && !self._framed(prepared) && !self._framed(cur.spec) &&
                    self._structureKey(prepared) === self._structureKey(cur.spec) &&
                    self._swapData(id, view, prepared)) {
                cur.spec = prepared;
                delete self._liveRows[id];
                self._ribbonViewControls(id, el);
            } else {
                self._embed(id, shown, opts, false);
            }
            self._origSpecs[id] = pristine;
            self._dataChanged(id);
        });
    },

    // A spec's JSON with its inline data rows blanked: equal keys mean two
    // specs compile to the same dataflow, differing only in the rows.
    _structureKey: function(spec) {
        return JSON.stringify(spec, function(k, v) {
            if (k === 'data' && v && typeof v === 'object' && Array.isArray(v.values)) {
                var shape = Object.assign({}, v);
                shape.values = null;
                return shape;
            }
            if (k === 'datasets' && v && typeof v === 'object') return Object.keys(v).sort();
            return v;
        });
    },

    // Insert `spec`'s inline datasets into the live view in place of its
    // current rows. `spec` matches the embedded one up to data rows, so its
    // compiled dataset names are the view's. Returns false (the caller then
    // re-embeds) when that cannot be established or a legend would be left
    // with an empty domain (handled by the embed-time legend drop).
    _swapData: function(id, view, spec) {
        var vg, self = this, opts = self._embedOpts[id] || {};
        try {
            vg = vegaLite.compile(JSON.parse(JSON.stringify(spec)),
                {logger: vega.logger(vega.None)}).spec;
            if (typeof opts.patch === 'function') vg = opts.patch(vg);
        } catch (e) { return false; }
        if (self._emptyLegends(vg).length) return false;
        var sources = (vg.data || []).filter(function(d) { return Array.isArray(d.values); });
        try { sources.forEach(function(d) { view.data(d.name); }); } catch (e) { return false; }
        var specRows = self._specRows[id] = self._specRows[id] || {};
        sources.forEach(function(d) {
            var rows = d.format ? vega.read(d.values, d.format) : d.values;
            specRows[d.name] = rows;
            view.change(d.name, vega.changeset().remove(function() { return true; }).insert(rows));
        });
        self._runRelayout(view);
        return true;
    },

    // Captioned figures render their raw-data / pretty-summary panes once,
    // on first open. After a data change, re-render the open ones and mark
    // the rest stale so they render fresh when opened.
    _dataChanged: function(id) {
        var self = this;
        self.whenReady(id, function() {
            var sel = '[data-aov-plot-id="' + id + '"]';
            document.querySelectorAll('.aov-data-raw-body' + sel + ', .aov-data-pretty-body' + sel)
                .forEach(function(body) {
                    if (body.dataset.aovRendered !== '1') return;
                    delete body.dataset.aovRendered;
                    var details = body.closest('details');
                    if (details && details.open) self._lazyRenderDataView(details,
                        body.classList.contains('aov-data-raw-body') ? 'raw' : 'pretty');
                });
        });
    },

    // Vega renders a legend whose scale domain is empty as a zero-item
    // legend group with inverted bounds; the legend layout folds those
    // bounds into the view origin, collapsing the whole canvas to 0x0
    // (observed on vega 5.33.1 / vega-lite 5.23.0). The incremental-plot
    // pattern starts from a typed-empty table, so this is the normal first
    // paint for any colored plot whose data streams in later. _withLiveRows
    // drops such legends at the compiled-spec boundary (an empty legend has
    // nothing to label), records the drop per plot, and the first data
    // change re-embeds once with live rows to restore the legend.
    _legendDomainEmpty: function(vg, domainSpec) {
        var self = this;
        if (Array.isArray(domainSpec)) return domainSpec.length === 0;
        if (domainSpec && domainSpec.fields) {
            // Multi-layer union domain: empty only when every field's
            // source dataset resolves empty.
            return (domainSpec.fields || []).every(function(f) {
                return self._legendDomainEmpty(vg, f);
            });
        }
        if (domainSpec && domainSpec.data) {
            var ds = (vg.data || []).find(function(d) { return d.name === domainSpec.data; });
            if (ds) return self._datasetEmpty(vg, ds);
        }
        return false;
    },
    _datasetEmpty: function(vg, ds) {
        if (ds.values) return ds.values.length === 0;
        if (ds.source) {
            var src = (vg.data || []).find(function(d) { return d.name === ds.source; });
            return src ? this._datasetEmpty(vg, src) : false;
        }
        return false;
    },
    _emptyLegends: function(vg) {
        var self = this;
        var scales = {};
        (vg.scales || []).forEach(function(s) { scales[s.name] = s; });
        return (vg.legends || []).filter(function(L) {
            return ['fill', 'stroke', 'shape', 'size', 'opacity', 'fontWeight'].some(function(prop) {
                var sc = L[prop] && scales[L[prop]];
                return sc && self._legendDomainEmpty(vg, sc.domain);
            });
        });
    },
    _dropEmptyLegends: function(id, vg) {
        var self = this, empty = self._emptyLegends(vg);
        var dropped = empty.length;
        vg.legends = (vg.legends || []).filter(function(L) { return empty.indexOf(L) === -1; });
        if (dropped > 0) self._droppedLegends[id] = dropped;
        else delete self._droppedLegends[id];
    },
    _maybeRestoreLegends: function(id) {
        var self = this;
        if (!self._droppedLegends[id]) return;
        delete self._droppedLegends[id];
        // keepData re-embed: the patched patch substitutes the appended
        // rows, so the legend comes back bound to the real domain. Signal
        // listeners re-attach in _embed; the view swap is invisible at
        // 60 fps (single bounded re-embed after the first data change).
        self._embed(id, self._origSpecs[id], self._embedOpts[id], true);
    },

    // A layer drawn with `visual(...; inspectable=false)` lets the pointer
    // through: Vega picks the mark beneath it for tooltips, aov:click,
    // aov:hover, touch and link groups. Vega-Lite has no mark property for
    // Vega's `interactive`, so the server styles the layer's marks
    // `aov-passive` and each compiled spec gets `interactive: false` on them
    // here (`_withLiveRows`, the cell charts). LOCKSTEP: `_AOV_PASSIVE`
    // (src/to_vegalite.jl) and the bare
    // `to_html(spec)` embed's `_PASSIVE_PATCH_JS` (src/to_html.jl).
    _passiveMarks: function(vg) {
        (function walk(marks) {
            (marks || []).forEach(function(m) {
                var style = m.style;
                if (style === 'aov-passive' || (Array.isArray(style) && style.indexOf('aov-passive') !== -1)) {
                    m.interactive = false;
                }
                walk(m.marks);
            });
        })(vg && vg.marks);
        return vg;
    },

    // AoV's automatic pan/zoom (the `grid` selection bound to the scales,
    // `add_auto_interactivity!`) makes Vega-Lite clip every mark of its view
    // to the frame. A plot then drew less than its `vsave` file before anyone
    // zoomed: text outside the frame (labels above a schedule's lines, a
    // direct label past the right edge) was cut, and `autosize: pad` reserved
    // no room for it, so an authored `padding` stayed blank. Such a mark is
    // drawn whole until the reader zooms or pans, and is then cut at the frame
    // only along the axes the selection moves: an x-only zoom cuts at the left
    // and right edges and keeps what lies above and below the frame. The
    // reader has zoomed while the top-level `grid` signal holds a domain for a
    // projected field, which is exactly what the zoomed scale's domain reads
    // (`grid["<field>"]`, in layered, faceted and concatenated views alike; a
    // facet's store stays empty). A double-click reset clears it and draws
    // everything again. The size of a mark's frame is its group's (`width` at
    // the top, `child_width` in a facet cell, ...); a mark in a group whose
    // size is not a signal or a number is cut at the whole frame once zoomed.
    // A layer that authors `clip: true` keeps it: its marks carry
    // `aov-own-clip`. LOCKSTEP: `_AOV_OWN_CLIP` (src/to_vegalite.jl) and the
    // bare `to_html(spec)` embed's `_PAN_ZOOM_CLIP_JS` (src/to_html.jl).
    _panZoomClip: function(vg) {
        if (!vg || !(vg.signals || []).some(function(s) { return s.name === 'grid'; })) return vg;
        function size(v) {
            return v && typeof v.signal === 'string' ? '(' + v.signal + ')' : typeof v === 'number' ? String(v) : null;
        }
        (function walk(group, w, h, fields) {
            (group.signals || []).forEach(function(s) {
                if (s.name === 'grid_tuple_fields' && Array.isArray(s.value)) fields = s.value;
            });
            (group.marks || []).forEach(function(m) {
                if (m.type === 'group') {
                    var u = (m.encode && m.encode.update) || {};
                    walk(m, size(u.width), size(u.height), fields);
                    return;
                }
                if (m.clip !== true || !fields || !fields.length) return;
                if ([].concat(m.style || []).indexOf('aov-own-clip') !== -1) return;
                var channels = fields.map(function(f) { return f.channel; });
                var x = channels.indexOf('x') !== -1, y = channels.indexOf('y') !== -1;
                var zoomed = fields.map(function(f) { return 'isValid(grid[' + JSON.stringify(f.field) + '])'; }).join(' || ');
                var cut = x && !y && w ? "pathShape('M0,-100000H' + " + w + " + 'V100000H0Z')" :
                    y && !x && h ? "pathShape('M-100000,0H100000V' + " + h + " + 'H-100000Z')" : 'true';
                m.clip = {signal: '(' + zoomed + ') ? ' + cut + ' : false'};
            });
        })(vg, 'width', 'height', null);
        return vg;
    },

    // The raw rows of datasets changed via updateData/appendData are kept per plot, and
    // re-embeds (resize, remapEncoding) compile the spec with them in place of its own.
    _withLiveRows: function(id, opts, gen) {
        var self = this, patch = opts.patch;
        return Object.assign({}, opts, {patch: function(vg) {
            if (typeof patch === 'function') vg = patch(vg);
            self._passiveMarks(vg);
            self._panZoomClip(vg);
            // An embed superseded or disposed while compiling still compiles;
            // it must not re-create the plot's state.
            if (self._gens[id] !== gen) return vg;
            var live = self._liveRows[id] || {}, specRows = self._specRows[id] = {};
            (vg.data || []).forEach(function(d) {
                if (Array.isArray(d.values)) specRows[d.name] = d.values;
                if (live[d.name]) d.values = live[d.name];
            });
            self._dropEmptyLegends(id, vg);
            return vg;
        }});
    },

    onSignal: function(id, signal, callback) {
        // A plot disposed before its embed resolved (to_node wires signals
        // in the embed's .then) has nothing to listen to. Listeners added
        // before a plot's first embed were already dropped by that embed.
        if (!(id in this._els)) return;
        // Kept per plot so re-embeds (resize, new layers) re-attach it
        this._signals[id] = this._signals[id] || [];
        this._signals[id].push({signal: signal, callback: callback});
        var view = this.views[id];
        if (view) this._attachSignal(view, signal, callback);
    },

    _attachSignal: function(view, signal, callback) {
        try {
            view.addSignalListener(signal, function(name, value) {
                callback(name, value, view);
            });
        } catch (e) { console.warn('AoV: cannot listen to signal', signal, e); }
    },

    // --- Frames: config(frame=…) (src/frame.jl) ---
    // A framed spec carries `_aov.frame` = {field, values, temporal,
    // interval (ms), loop, start} and a Vega-Lite parameter `aov_frame`, the
    // shown frame's value, which the first transform of every unit filters
    // on. The runtime owns the player: the play/pause button, slider and
    // frame label after the plot, the timer, and the shown frame, kept per
    // plot across re-embeds (resize, theme, picker, update_spec).
    _FRAME_PARAM: 'aov_frame',
    _framed: function(spec) { return !!(spec && spec._aov && spec._aov.frame); },
    // Set up (or drop) the player for the spec an embed is about to show;
    // returns the promise of its frame plan, or null for an unframed spec.
    _frameSetup: function(id, el, spec, opts) {
        var meta = spec && spec._aov && spec._aov.frame;
        if (!meta) {
            this._frameStop(id);
            var old = this._frames[id];
            if (old && old.box) old.box.remove();
            delete this._frames[id];
            return null;
        }
        var st = this._frames[id];
        if (!st) {
            st = this._frames[id] = {index: meta.start || 0, playing: false, timer: null, box: null};
        } else if (st.values) {
            // The same plot with a new spec: keep showing the same frame when
            // it is still one.
            var at = this._frameIndex(meta.values, st.values[st.index]);
            st.index = at !== -1 ? at : Math.min(st.index, meta.values.length - 1);
        }
        st.values = meta.values;
        st.meta = meta;
        this._frameControls(id, el);
        return this._framePlan(id, spec, opts);
    },
    _frameIndex: function(values, value) {
        var key = JSON.stringify(value);
        for (var i = 0; i < values.length; i++) if (JSON.stringify(values[i]) === key) return i;
        return -1;
    },
    // The scale domains and map projection fits of all frames together. The
    // compiled spec runs headless once per frame; each data-driven scale
    // domain, and the features of each fitted projection, are unioned. So a
    // colour means the same value in every frame, axes do not jump and a map
    // keeps its framing, whatever the marks compute (stacked bars, Vega-Lite
    // density or regression transforms). Only the data runs per frame
    // (_framePlanView), not the marks. A spec with neither a data-driven scale
    // nor a fitted projection needs no run at all. Resolves to null when it
    // cannot be computed; the frames then re-fit their scales one by one.
    _framePlan: function(id, spec, opts) {
        var self = this, meta = spec._aov.frame, view = null, P = this._FRAME_PARAM;
        function dataDriven(d) { return !!d && typeof d === 'object' && !Array.isArray(d) && !d.signal; }
        function discrete(t) { return t === 'ordinal' || t === 'band' || t === 'point'; }
        function token(v) { return v instanceof Date ? 'd' + v.getTime() : typeof v + ':' + v; }
        // The values of `dom` not yet in `acc`, each inserted after its
        // predecessor in `dom`: frames that order their values consistently
        // merge in that order.
        function merge(acc, dom) {
            var keys = acc.map(token), at = -1;
            dom.forEach(function(v) {
                var i = keys.indexOf(token(v));
                if (i === -1) { at++; acc.splice(at, 0, v); keys.splice(at, 0, token(v)); }
                else at = i;
            });
            return acc;
        }
        return Promise.resolve().then(function() {
            var s = JSON.parse(JSON.stringify(spec));
            // Domains do not depend on the size; a headless view has no container.
            delete s.autosize;
            if (s.width === 'container') s.width = 400;
            var vg = vegaLite.compile(s, {logger: vega.logger(vega.None)}).spec;
            if (!(vg.signals || []).some(function(sig) { return sig.name === P; })) return null;
            var scales = (vg.scales || []).filter(function(sc) { return dataDriven(sc.domain); });
            var fitted = (vg.projections || []).map(function(p) { return !!(p.fit && p.fit.signal); });
            if (!scales.length && fitted.indexOf(true) === -1) return {domains: {}, fits: []};
            vg.signals = vg.signals.concat((vg.projections || []).filter(function(p, j) { return fitted[j]; })
                .map(function(p) { return {name: '__aov_frame_fit_' + vg.projections.indexOf(p), update: p.fit.signal}; }));
            view = new vega.View(vega.parse(self._framePlanView(vg)), {renderer: 'none', loader: opts && opts.loader || undefined});
            var domains = {}, fits = fitted.map(function() { return {seen: new Set(), points: {}, features: []}; });
            var collect = function() {
                scales.forEach(function(sc) {
                    var scale = view.scale(sc.name);
                    if (!scale) return;
                    var d = scale.domain(), acc = domains[sc.name];
                    if (discrete(sc.type)) { domains[sc.name] = merge(acc || [], d); return; }
                    if (d.length !== 2) return;
                    var lo = +d[0], hi = +d[1];
                    if (!isFinite(lo) || !isFinite(hi)) return;
                    domains[sc.name] = acc ? [Math.min(acc[0], lo), Math.max(acc[1], hi)] : [lo, hi];
                });
                fitted.forEach(function(on, j) {
                    if (!on) return;
                    var f = fits[j];
                    (function add(x) {
                        if (!x) return;
                        if (Array.isArray(x)) { x.forEach(add); return; }
                        if (x.type === 'FeatureCollection') { (x.features || []).forEach(add); return; }
                        if (f.seen.has(x)) return;
                        f.seen.add(x);
                        // Point features are rebuilt every run: one per position.
                        if (x.geometry && x.geometry.type === 'Point') {
                            var key = JSON.stringify(x.geometry.coordinates);
                            if (f.points[key]) return;
                            f.points[key] = true;
                        }
                        f.features.push(x);
                    })(view.signal('__aov_frame_fit_' + j));
                });
            };
            var i = 0;
            var next = function() {
                if (i >= meta.values.length) return;
                view.signal(P, meta.values[i++]);
                return view.runAsync().then(function() { collect(); return next(); });
            };
            return next().then(function() {
                scales.forEach(function(sc) {
                    var d = domains[sc.name];
                    if (d && discrete(sc.type) && sc.domain.sort === true) d.sort(vega.ascending);
                });
                return {domains: domains, fits: fits.map(function(f) {
                    return f.features.length ? {type: 'FeatureCollection', features: f.features} : null;
                })};
            });
        }).catch(function(err) {
            console.error('AoV: plot "' + id + '": could not compute the scales its frames share, so each frame fits its own:', err);
            return null;
        }).then(function(plan) {
            if (view) view.finalize();
            return plan;
        });
    },
    // The compiled spec `vg` (the frame plan's own copy, changed in place)
    // cut down to what the plan reads: the top-level scale domains and the
    // features each fitted projection fits. Marks, axes, legends and titles
    // feed neither, and neither does a projection's fit itself, which only
    // places marks. Kept, they were the plan's whole cost: a map's outlines
    // went through the projection twice per frame, for the fit and for the
    // marks' bounds (snag `frame-plan-repla-b3f8cee5`: 145 frames of 16 states
    // with 28,902 vertices took 3 s, the data alone 0.1 s). A geopoint
    // transform (lon/lat marks on a map) only adds the marks' pixel
    // positions, so it goes too unless something left reads them; then a
    // projection nothing left names loses its fit. Anything else that names
    // a projection keeps it fitted, so what it computes is unchanged.
    _framePlanView: function(vg) {
        // Whether the JSON `text` mentions `name` as a string (a field, a
        // projection), inside an expression too: datum["f"] (escaped there
        // as datum[\"f\"]), 'f', datum.f.
        function names(text, name) {
            var q = String(name).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            return new RegExp('\\\\?"' + q + '\\\\?"|\'' + q + '\'|datum\\.' + q + '\\b').test(text);
        }
        delete vg.marks;
        delete vg.axes;
        delete vg.legends;
        delete vg.title;
        (vg.data || []).forEach(function(d) {
            (d.transform || []).slice().forEach(function(t) {
                if (t.type !== 'geopoint') return;
                var rest = JSON.stringify(vg, function(k, v) { return v === t ? undefined : v; });
                if (!(t.as || ['x', 'y']).some(function(f) { return names(rest, f); })) d.transform.splice(d.transform.indexOf(t), 1);
            });
        });
        var rest = JSON.stringify(Object.assign({}, vg, {projections: undefined}));
        (vg.projections || []).forEach(function(p) { if (p.fit && !names(rest, p.name)) delete p.fit; });
        return vg;
    },
    // `opts` whose compiled spec shows the current frame with the plan's
    // shared domains and projection fits.
    _withFrame: function(id, opts, plan) {
        var self = this, st = this._frames[id];
        if (!st) return opts;
        var patch = opts.patch;
        return Object.assign({}, opts, {patch: function(vg) {
            if (typeof patch === 'function') vg = patch(vg);
            var value = st.values[st.index];
            (vg.signals || []).forEach(function(sig) { if (sig.name === self._FRAME_PARAM) sig.value = value; });
            if (!plan) return vg;
            (vg.scales || []).forEach(function(sc) {
                var d = plan.domains[sc.name];
                if (d && sc.domain && typeof sc.domain === 'object' && !Array.isArray(sc.domain) && !sc.domain.signal) {
                    sc.domain = d.slice();
                }
            });
            (vg.projections || []).forEach(function(p, j) {
                if (!plan.fits[j] || !(p.fit && p.fit.signal)) return;
                var name = '__aov_frame_fit_' + j;
                vg.signals = (vg.signals || []).concat([{name: name, value: plan.fits[j]}]);
                p.fit = {signal: name};
            });
            return vg;
        }});
    },
    // A view embedded while the shown frame changed (a click during an
    // embed) catches up with it.
    _frameSync: function(id, view) {
        var st = this._frames[id];
        if (!st) return;
        var want = st.values[st.index];
        try {
            if (JSON.stringify(view.signal(this._FRAME_PARAM)) !== JSON.stringify(want)) {
                view.signal(this._FRAME_PARAM, want);
                view.runAsync();
            }
        } catch (e) { console.error('[' + id + '] frame', e); }
    },
    // The player after the plot (after its scroll frame, so it never scrolls
    // away with a wide plot). One element per plot id, reused by a same-id
    // re-render; its listeners look the plot up by id.
    _frameControls: function(id, el) {
        var self = this, st = this._frames[id];
        var host = el && el.parentElement && el.parentElement.classList.contains('aov-plot-area') ? el.parentElement : el;
        if (!host || !host.parentNode) return;
        var box = host.nextElementSibling;
        if (!box || box.getAttribute('data-aov-frame-for') !== id) {
            box = document.createElement('div');
            box.className = 'aov-frame';
            box.setAttribute('data-aov-frame-for', id);
            // The layout also holds on runtime-only pages, which load no aov.css.
            box.style.display = 'flex';
            box.style.alignItems = 'center';
            box.style.gap = '0.5em';
            var play = document.createElement('button');
            play.type = 'button';
            play.className = 'aov-frame-play';
            var slider = document.createElement('input');
            slider.type = 'range';
            slider.className = 'aov-frame-slider';
            slider.min = '0';
            slider.step = '1';
            var out = document.createElement('output');
            out.className = 'aov-frame-value';
            box.appendChild(play);
            box.appendChild(slider);
            box.appendChild(out);
            play.addEventListener('click', function() {
                var cur = self._frames[id];
                if (cur && cur.playing) self.pauseFrames(id); else self.playFrames(id);
            });
            slider.addEventListener('input', function() {
                var index = +slider.value;
                self.pauseFrames(id);
                self.setFrame(id, index);
            });
            host.insertAdjacentElement('afterend', box);
        }
        st.box = box;
        var range = box.querySelector('.aov-frame-slider');
        range.max = String(st.values.length - 1);
        range.setAttribute('aria-label', st.meta.field);
        this._frameRender(id);
    },
    // A frame value as the player shows it: a midnight timestamp as its date.
    _frameLabel: function(v) {
        var m = typeof v === 'string' && /^(\d{4}-\d{2}-\d{2})T00:00(:00(\.0+)?)?$/.exec(v);
        return m ? m[1] : String(v);
    },
    _frameRender: function(id) {
        var st = this._frames[id];
        if (!st || !st.box) return;
        var text = this._texts(st.box), word = text(st.playing ? 'pause' : 'play');
        var play = st.box.querySelector('.aov-frame-play');
        var range = st.box.querySelector('.aov-frame-slider');
        var out = st.box.querySelector('.aov-frame-value');
        play.textContent = st.playing ? '⏸︎' : '▶︎';
        play.setAttribute('aria-label', word);
        play.title = word;
        var label = this._frameLabel(st.values[st.index]);
        range.value = String(st.index);
        range.setAttribute('aria-valuetext', label);
        out.textContent = label;
    },
    // Public: show frame `index` (0-based) of framed plot `id`.
    setFrame: function(id, index) {
        var st = this._frames[id];
        if (!st) return Promise.resolve();
        st.index = Math.max(0, Math.min(st.values.length - 1, Math.round(+index) || 0));
        this._frameRender(id);
        var view = this.views[id];
        if (!view) return Promise.resolve();
        try { view.signal(this._FRAME_PARAM, st.values[st.index]); }
        catch (e) { console.error('[' + id + '] frame', e); return Promise.resolve(); }
        return view.runAsync();
    },
    // Public: step through the frames, one every `interval` ms; from the
    // last frame, play starts over at the first. Without `loop` it stops
    // on the last frame.
    playFrames: function(id) {
        var self = this, st = this._frames[id];
        if (!st || st.playing) return;
        st.playing = true;
        if (st.index >= st.values.length - 1) self.setFrame(id, 0); else self._frameRender(id);
        var tick = function() {
            if (self._frames[id] !== st || !st.playing) return;
            var next = st.index + 1;
            if (next >= st.values.length) {
                if (!st.meta.loop) { self.pauseFrames(id); return; }
                next = 0;
            }
            var again = function() {
                if (self._frames[id] === st && st.playing) st.timer = setTimeout(tick, st.meta.interval);
            };
            self.setFrame(id, next).then(again, again);
        };
        st.timer = setTimeout(tick, st.meta.interval);
    },
    // Public: stop playing; the shown frame stays.
    pauseFrames: function(id) {
        this._frameStop(id);
        this._frameRender(id);
    },
    _frameStop: function(id) {
        var st = this._frames[id];
        if (!st) return;
        st.playing = false;
        if (st.timer) clearTimeout(st.timer);
        st.timer = null;
    },
    _refuseFramed: function(fn, id) {
        var cur = this._cur[id];
        if (!cur || !this._framed(cur.spec)) return false;
        console.error('AoV.' + fn + ': plot "' + id + '" is a frame animation (config(frame=…)), whose ' +
            'frames and shared scales come from all of its rows, so rows cannot be changed in place: send ' +
            'update_spec(id, spec) with the new rows instead. Data left unchanged.');
        return true;
    },

    // --- Cell charts: cell_charts(spec) (src/cell_charts.jl) ---
    // A table column of small charts is a figure stepped one frame per table
    // row key (config(frame=…) lowering). `cellColumn` registers the column's
    // figure once per page; each cell (`AoV.cell(el)`: el carries the column
    // id in data-aov-cells and its key as JSON in data-aov-key) gets its own
    // light vega.View of the column's Vega, parsed once per column and host
    // theme, fed only its own rows. The frame plan pins every scale to what
    // all keys share, except the axes the figure unlinks (`_aov.cells.own`).
    // Cells near the viewport draw first; the rest draw while the browser is
    // idle, so a printed page has them all. Cells are no plots: their views
    // live in _cellViews (element → {view, colId, theme}), not in the
    // per-plot maps, and the removal sweep finalizes the view of a cell that
    // left the page.
    _cellCols: {},
    _cellWaiting: {},
    _cellViews: null,
    _cellPending: null,
    cellColumn: function(colId, spec) {
        var self = this;
        spec = self._expandSpecData(JSON.parse(JSON.stringify(spec)));
        var known = self._cellCols[colId];
        // The same column sent again with a re-rendered row keeps its parsed Vega.
        if (!known || JSON.stringify(known.spec) !== JSON.stringify(spec)) {
            self._cellCols[colId] = {spec: spec, plan: null, built: {}, inline: null};
        }
        var waiting = self._cellWaiting[colId] || [];
        delete self._cellWaiting[colId];
        waiting.forEach(function(el) { self._drawCell(el); });
    },
    cell: function(el) {
        var self = this;
        if (!el) return;
        self._watchRemovals();
        self._watchTheme();
        if (!self._cellViews) {
            self._cellViews = new Map();
            self._cellPending = new Set();
            if (typeof IntersectionObserver !== 'undefined') {
                self._cellIO = new IntersectionObserver(function(entries) {
                    entries.forEach(function(e) { if (e.isIntersecting) self._takeCell(e.target); });
                }, {rootMargin: '100% 0px'});
            }
            window.addEventListener('beforeprint', function() { self._cellPending.forEach(function(c) { self._takeCell(c); }); });
            self._cellTotals = new Map();
            if (typeof ResizeObserver !== 'undefined') {
                self._cellRO = new ResizeObserver(function() { self._cellAlignSoon(); });
            }
            if (document.fonts && document.fonts.addEventListener) {
                document.fonts.addEventListener('loadingdone', function() { self._cellAlignSoon(); });
            }
        }
        self._cellPending.add(el);
        if (self._cellIO) self._cellIO.observe(el);
        self._cellFillSoon();
        var td = el.closest ? el.closest('td.aov-cells') : null;
        var total = td && td.querySelector(':scope > .aov-cell-total');
        if (total && !self._cellTotals.has(total)) {
            self._cellTotals.set(total, el.getAttribute('data-aov-cells'));
            if (self._cellRO) self._cellRO.observe(total);
        }
        self._cellAlignSoon();
    },
    // A column's totals share one slot, as wide as its widest total (the
    // stylesheet right-aligns each total in it with equal-width digits), so
    // every chart of the column starts at the same x, whatever the totals'
    // lengths and the page's font. A column is a column id within one
    // table. Measured in a task of its own: after cells arrive or leave, and
    // when a total changes size (a hidden table shown, web fonts loaded).
    // A table not laid out yet measures 0 and keeps its slots until then.
    _cellAlignSoon: function() {
        var self = this;
        if (self._cellAligning) return;
        self._cellAligning = setTimeout(function() { self._cellAligning = 0; self._cellAlign(); }, 0);
    },
    _cellAlign: function() {
        var columns = new Map();
        this._cellTotals.forEach(function(colId, total) {
            if (!total.isConnected) return;
            var table = total.closest('table'), byId = columns.get(table);
            if (!byId) columns.set(table, byId = {});
            (byId[colId] = byId[colId] || []).push(total);
        });
        var groups = [];
        columns.forEach(function(byId) { Object.keys(byId).forEach(function(id) { groups.push(byId[id]); }); });
        // Clear every slot, read every width, then set: one layout for all columns.
        groups.forEach(function(totals) { totals.forEach(function(t) { t.style.minWidth = ''; }); });
        var widths = groups.map(function(totals) {
            return totals.reduce(function(w, t) { return Math.max(w, t.getBoundingClientRect().width); }, 0);
        });
        groups.forEach(function(totals, i) {
            if (widths[i] > 0) totals.forEach(function(t) { t.style.minWidth = Math.ceil(widths[i]) + 'px'; });
        });
    },
    _takeCell: function(el) {
        if (!this._cellPending.delete(el)) return;
        if (this._cellIO) this._cellIO.unobserve(el);
        this._drawCell(el);
    },
    // Draw the cells no one has scrolled to yet, a few per idle period.
    _cellFillSoon: function() {
        var self = this;
        if (self._cellFilling) return;
        self._cellFilling = true;
        var idle = window.requestIdleCallback ? function(fn) { window.requestIdleCallback(fn, {timeout: 1000}); } :
            function(fn) { setTimeout(fn, 50); };
        idle(function step() {
            var batch = [];
            self._cellPending.forEach(function(el) { if (batch.length < 8) batch.push(el); });
            batch.forEach(function(el) { self._takeCell(el); });
            if (self._cellPending.size) idle(step); else self._cellFilling = false;
        });
    },
    _drawCell: function(el) {
        var self = this;
        if (!el.isConnected) return;
        var colId = el.getAttribute('data-aov-cells'), col = self._cellCols[colId];
        if (!col) { (self._cellWaiting[colId] = self._cellWaiting[colId] || []).push(el); return; }
        var key;
        try { key = JSON.parse(el.getAttribute('data-aov-key')); }
        catch (err) { console.error('AoV: a cell of column "' + colId + '" has no readable key:', err); return; }
        if (!col.plan) col.plan = self._framePlan(colId, col.spec);
        var tip = self._themeOn(col.spec) ? self._tooltipTheme(el) : 'light';
        return Promise.all([col.plan, self._cellTooltip(tip)]).then(function(r) {
            return self._cellView(col, colId, el, key, r[0], r[1]);
        });
    },
    // The column's Vega for the host theme of `el`, parsed once: compiled as
    // vega-embed compiles a plot (the theme as the config the spec's own
    // config overrides), with the shared domains of `plan` pinned and its
    // inline datasets emptied. Their rows are kept per key (`col.inline`),
    // each with the rows that carry no key, which every cell draws.
    _cellRuntime: function(col, el, plan) {
        var spec = col.spec;
        var theme = this._themeOn(spec) ? this.hostThemeConfig(el) : null;
        var tkey = JSON.stringify(theme);
        if (col.built[tkey]) return col.built[tkey];
        var s = JSON.parse(JSON.stringify(spec));
        if (theme) this._hostInk(s, theme);
        var vg = this._passiveMarks(vegaLite.compile(s, theme ? {config: theme} : {}).spec);
        var own = (spec._aov && spec._aov.cells && spec._aov.cells.own) || [];
        if (plan) (vg.scales || []).forEach(function(sc) {
            var d = plan.domains[sc.name];
            if (!d || own.indexOf(sc.name) !== -1) return;
            if (sc.domain && typeof sc.domain === 'object' && !Array.isArray(sc.domain) && !sc.domain.signal) sc.domain = d.slice();
        });
        var field = spec._aov.frame.field, inline = [];
        (vg.data || []).forEach(function(d) {
            if (!Array.isArray(d.values)) return;
            var byKey = new Map(), shared = [];
            d.values.forEach(function(row) {
                var k = row && row[field];
                if (k === undefined || k === null || (typeof k === 'number' && !isFinite(k))) { shared.push(row); return; }
                var tok = JSON.stringify(k);
                if (!byKey.has(tok)) byKey.set(tok, []);
                byKey.get(tok).push(row);
            });
            inline.push({name: d.name, byKey: byKey, shared: shared});
            d.values = [];
        });
        col.inline = inline;
        return (col.built[tkey] = {runtime: vega.parse(vg), theme: tkey});
    },
    // vega-embed's tooltip handler for `theme`, the one every AoV plot in that
    // look shows: vega-embed exports no constructor for it, so a view embedded
    // once into a detached element hands it over. A cell's view takes it when
    // it is built: setting a handler on a drawn view resets its renderer and
    // leaves it blank.
    _cellTooltip: function(theme) {
        if (!this._cellTips[theme]) {
            this._cellTips[theme] = vegaEmbed(document.createElement('div'), {},
                    {mode: 'vega', renderer: 'none', actions: false, tooltip: {theme: theme}})
                .then(function(r) { var handler = r.view.tooltip(); r.finalize(); return handler; })
                .catch(function(err) { console.error('AoV: cell charts show no tooltips:', err); return null; });
        }
        return this._cellTips[theme];
    },
    _cellView: function(col, colId, el, key, plan, tooltip) {
        var self = this;
        if (!el.isConnected) return;
        var built;
        try { built = self._cellRuntime(col, el, plan); }
        catch (err) { console.error('AoV: cell charts of column "' + colId + '" cannot be drawn:', err); return; }
        var prev = self._cellViews.get(el);
        if (prev) prev.view.finalize();
        var opts = {renderer: 'svg', container: el, hover: true};
        if (tooltip) opts.tooltip = tooltip;
        var view = new vega.View(built.runtime, opts);
        var tok = JSON.stringify(key);
        col.inline.forEach(function(ds) { view.data(ds.name, (ds.byKey.get(tok) || []).concat(ds.shared)); });
        view.signal(self._FRAME_PARAM, key);
        self._cellViews.set(el, {view: view, colId: colId, theme: built.theme});
        return view.runAsync().then(function() { return view; })
            .catch(function(err) { console.error('AoV: a cell of column "' + colId + '" cannot be drawn:', err); });
    },
    // Redraw the cells of every column whose host theme changed (refreshTheme).
    _refreshCellThemes: function() {
        var self = this;
        if (!self._cellViews) return;
        var current = {};
        self._cellViews.forEach(function(c, el) {
            var col = self._cellCols[c.colId];
            if (!col || !el.isConnected) return;
            if (!(c.colId in current)) current[c.colId] = JSON.stringify(self._themeOn(col.spec) ? self.hostThemeConfig(el) : null);
            if (current[c.colId] !== c.theme) self._drawCell(el);
        });
    },
    // Release the views of cells that left the page (the removal sweep).
    _sweepCells: function() {
        var self = this;
        if (!self._cellViews) return;
        self._cellViews.forEach(function(c, el) {
            if (el.isConnected) return;
            c.view.finalize();
            self._cellViews.delete(el);
        });
        self._cellPending.forEach(function(el) {
            if (el.isConnected) return;
            if (self._cellIO) self._cellIO.unobserve(el);
            self._cellPending.delete(el);
        });
        // A row that left can take its column's widest total with it.
        var left = false;
        self._cellTotals.forEach(function(colId, total) {
            if (total.isConnected) return;
            if (self._cellRO) self._cellRO.unobserve(total);
            self._cellTotals.delete(total);
            left = true;
        });
        if (left) self._cellAlignSoon();
    },

    // A click on a data mark reaches app code as a bubbling `aov:click`
    // event on the plot element, detail {id, datum, event}: listen with
    // `document.addEventListener('aov:click', …)` or htmx
    // `hx-trigger="aov:click from:#<id>"`. Every view this runtime embeds
    // gets the listener, so it survives resize, picker, theme and
    // update_spec re-embeds. Legend, axis and title items have their own
    // roles, and an empty facet cell is a group ('scope'): none of them is
    // reported. A nearest-point overlay item's datum is the scenegraph item
    // of the point under it, whose own datum is the row. On touch input a
    // long press is the click and a tap is not (see _attachTouch), so the
    // click a browser sends after a touch is not reported here.
    _attachClick: function(id, view) {
        var self = this;
        view.addEventListener('click', function(event, item) {
            if (self._fromTouch(id, event)) return;
            if (!item || !item.mark || item.mark.role !== 'mark') return;
            var datum = item.datum;
            while (datum && datum.mark && datum.mark.marktype && datum.datum) datum = datum.datum;
            var el = self._els[id];
            if (!datum || !el) return;
            el.dispatchEvent(new CustomEvent('aov:click', {bubbles: true,
                detail: {id: id, datum: Object.assign({}, datum), event: event}}));
        });
    },

    _dispatch: function(id, name, detail) {
        var el = this._els[id];
        if (el) el.dispatchEvent(new CustomEvent(name, {bubbles: true, detail: detail}));
    },
    // The row a data-mark item was drawn from, as `_attachClick` reads it;
    // null for legend, axis and title items and an empty facet cell.
    _markRow: function(item) {
        if (!item || !item.mark || item.mark.role !== 'mark') return null;
        var datum = item.datum;
        while (datum && datum.mark && datum.mark.marktype && datum.datum) datum = datum.datum;
        return datum || null;
    },

    // Hovering a data mark reaches app code as a bubbling `aov:hover` event
    // on the plot element, detail {id, datum, event}; leaving it sends
    // `aov:hoverend`, detail {id, datum, event}. Moving from one mark to the
    // next sends the old mark's hoverend, then the new mark's hover. The
    // marks are the ones `aov:click` reports: on a scatter with tooltips the
    // nearest point. Vega reports an item change as mouseout then mouseover
    // in one go, so a mouseout ends the hover only if no mouseover of the
    // same row follows it; leaving the plot element ends it at once, which
    // also covers a view replaced while the pointer was over it. A finger
    // hovers by a tap instead (see _attachTouch): a finger dragging over
    // the plot hovers nothing.
    _attachHover: function(id, view) {
        var self = this, el = self._els[id];
        view.addEventListener('mouseover', function(event, item) {
            var row = self._markRow(item);
            if (row && !self._fromTouch(id, event)) self._hoverStart(id, row, event);
        });
        view.addEventListener('mouseout', function(event, item) {
            if (self._markRow(item) && !self._fromTouch(id, event)) self._hoverLeave(id, event);
        });
        if (el && !el._aovHoverLeave) {
            el._aovHoverLeave = true;
            el.addEventListener('pointerleave', function(event) {
                if (self._els[id] === el && !self._fromTouch(id, event)) self._hoverEnd(id, event);
            });
        }
    },
    _hoverStart: function(id, row, event) {
        var h = this._hovers[id];
        if (h && h.row === row) { h.leaving = false; return; }
        if (h) this._hoverEnd(id, event);
        // Each event carries its own copy of the row, as aov:click does.
        h = this._hovers[id] = {row: row, leaving: false, key: null};
        this._dispatch(id, 'aov:hover', {id: id, datum: Object.assign({}, row), event: event});
        h.key = this._announce(id, row);
    },
    _hoverLeave: function(id, event) {
        var self = this, h = self._hovers[id];
        if (!h) return;
        h.leaving = true;
        Promise.resolve().then(function() {
            if (self._hovers[id] === h && h.leaving) self._hoverEnd(id, event);
        });
    },
    _hoverEnd: function(id, event) {
        var h = this._hovers[id];
        if (!h) return;
        delete this._hovers[id];
        this._dispatch(id, 'aov:hoverend', {id: id, datum: Object.assign({}, h.row), event: event});
        if (h.key !== null) this._announceKey(id, null);
    },

    // --- Touch input ---
    // A finger has no hover, and Vega picks a canvas mark only on
    // pointermove, which a tap never sends: a tap's click names no mark,
    // and a tooltip shows only while a finger drags over its mark, under
    // the finger. So the runtime reads touches itself, through Vega's
    // touchstart, for which Vega picks the item under the finger (canvas
    // and SVG alike):
    // - A tap on a data mark opens its tooltip above the finger and hovers
    //   its row (`aov:hover`, link groups), as a mouse over the mark does.
    //   Both stay after the finger lifts and follow the tapped spot when the
    //   page scrolls. They end (`aov:hoverend`) on a tap on empty plot
    //   space, a press outside the plot, a finger dragging on the plot, or
    //   once the tapped spot leaves the visible area; a tap on another mark
    //   moves them there.
    // - A long press on a data mark (_TOUCH_HOLD_MS, the finger within
    //   _TOUCH_SLOP px) opens them too and is the mark's click: one
    //   `aov:click`, detail.event the touchstart. The context menu over the
    //   mark and the click the browser sends on release are suppressed.
    // - A tap never sends `aov:click`, so the tooltip can be read before an
    //   app acts on the mark. Scrolling and pinch zoom that start on the
    //   plot keep working: no touchstart or touchmove is prevented, and the
    //   plot sets no touch-action.
    // Mouse input is unchanged, and so is a pen that hovers.
    _TOUCH_HOLD_MS: 500,
    _TOUCH_SLOP: 10,
    _TOUCH_TIP_GAP: 24,
    _attachTouch: function(id, view) {
        var self = this, t = self._touches[id];
        // A press or an open tooltip holds an item of the view a re-embed replaced.
        if (t && t.press && t.press.view !== view) self._touchCancel(id);
        if (t && t.tip && t.tip.view !== view) self._touchClose(id, null);
        view.addEventListener('touchstart', function(event, item) {
            self._touchStart(id, view, event, item);
        });
        self._attachTouchElement(id, self._els[id]);
    },
    // The plot element's own touch listeners, attached once per element and
    // kept across re-embeds. They are made here, apart from _attachTouch: the
    // closures one function call creates share one scope, so a listener made
    // beside the touchstart handler above would hold its `view`, and the
    // element would keep its first embed's view (data, dataflow, scenegraph)
    // reachable after every re-embed and after dispose (snag
    // attachtouch-clos-f2d7a0dc). A listener the element holds never closes
    // over a view; it reaches the plot's current state by id.
    _attachTouchElement: function(id, el) {
        var self = this;
        if (!el || el._aovTouch) return;
        el._aovTouch = true;
        var mine = function() { return self._els[id] === el; };
        el.addEventListener('touchmove', function(event) { if (mine()) self._touchMove(id, event); }, {passive: true});
        el.addEventListener('touchend', function(event) { if (mine()) self._touchEnd(id, event); });
        el.addEventListener('touchcancel', function() { if (mine()) self._touchCancel(id); }, {passive: true});
        el.addEventListener('contextmenu', function(event) { if (mine()) self._touchMenu(id, event); });
    },
    _touchStart: function(id, view, event, item) {
        var self = this, el = self._els[id], p = event.changedTouches && event.changedTouches[0];
        var t = self._touches[id] || (self._touches[id] = {press: null, tip: null, emulatedAt: 0, heldAt: 0});
        self._touchCancel(id);
        // A second finger (a pinch) presses nothing.
        if (!p || (event.touches && event.touches.length > 1)) return;
        if (el && !el._aovTouchStyled) {
            // No text selection or iOS callout over the plot on a long press.
            el._aovTouchStyled = true;
            el.style.setProperty('-webkit-touch-callout', 'none');
            el.style.setProperty('-webkit-user-select', 'none');
            el.style.setProperty('user-select', 'none');
        }
        var press = t.press = {view: view, item: item, row: self._markRow(item), x: p.clientX, y: p.clientY,
            event: event, held: false, timer: null};
        if (press.row) press.timer = setTimeout(function() {
            if (t.press === press) self._touchHold(id);
        }, self._TOUCH_HOLD_MS);
    },
    _touchMove: function(id, event) {
        var t = this._touches[id], press = t && t.press, p = event.changedTouches && event.changedTouches[0];
        if (!press) return;
        var dx = p ? p.clientX - press.x : 0, dy = p ? p.clientY - press.y : 0;
        if (p && event.touches.length < 2 && dx * dx + dy * dy <= this._TOUCH_SLOP * this._TOUCH_SLOP) return;
        // A finger dragging on the plot (a scroll, a pan, a pinch) presses
        // nothing and closes the plot's tooltip.
        this._touchCancel(id);
        this._touchClose(id, event);
    },
    _touchEnd: function(id, event) {
        var self = this, t = self._touches[id], press = t && t.press;
        if (!press) return;
        self._touchCancel(id);
        // A long press was the click, and a tap on a mark is none: the
        // browser sends neither its click nor the mouse events it emulates
        // after a touch, whose mouseout over a mark (SVG) would hide the
        // tooltip and end the hover again. A touchend the browser does not
        // let us cancel still gets them (see _fromTouch).
        if (press.held || press.row) {
            if (event.cancelable) event.preventDefault(); else t.emulatedAt = Date.now();
        }
        if (press.held) return;
        // A tap. It takes effect once the browser has sent the pointer
        // events that end the touch, whose mouseout hides a tooltip Vega
        // showed while the finger moved.
        setTimeout(function() {
            if (self._touches[id] !== t) return;
            if (press.row) self._touchOpen(id, press); else self._touchClose(id, event);
        }, 0);
    },
    _touchHold: function(id) {
        var t = this._touches[id], press = t && t.press, el = this._els[id];
        if (!press || press.held || !press.row) return;
        press.held = true;
        clearTimeout(press.timer);
        t.heldAt = Date.now();
        // As with a mouse, the clicked mark is the hovered one.
        this._touchOpen(id, press);
        if (el) el.dispatchEvent(new CustomEvent('aov:click', {bubbles: true,
            detail: {id: id, datum: Object.assign({}, press.row), event: press.event}}));
    },
    // No context menu over a pressed mark. A browser that opens it before
    // _TOUCH_HOLD_MS (Android's long press) makes it the long press; one
    // that opens it on release (after the hold) is prevented as well.
    _touchMenu: function(id, event) {
        var t = this._touches[id];
        if (!t) return;
        if (t.press && t.press.row) {
            event.preventDefault();
            this._touchHold(id);
        } else if (t.heldAt && Date.now() - t.heldAt < 1000) event.preventDefault();
    },
    _touchCancel: function(id) {
        var t = this._touches[id];
        if (!t || !t.press) return;
        clearTimeout(t.press.timer);
        t.press = null;
    },
    // Whether a pointer or mouse event comes from a finger: its pointerType
    // says so, or it is one of the mouse events (and the click) a browser
    // emulates after a touch it did not let _touchEnd cancel, which carry
    // no pointerType.
    _fromTouch: function(id, event) {
        var t = this._touches[id];
        if (event && event.pointerType === 'touch') return true;
        return !!(t && t.emulatedAt && Date.now() - t.emulatedAt < 1000);
    },
    _touchOpen: function(id, press) {
        var t = this._touches[id], el = this._els[id];
        if (!t || !el || this.views[id] !== press.view) return;
        var prev = t.tip, r = el.getBoundingClientRect();
        if (prev) prev.view._handler.handleTooltip({clientX: 0, clientY: 0}, prev.item, false);
        t.tip = {view: press.view, item: press.item, row: press.row, dx: press.x - r.left, dy: press.y - r.top};
        this._watchTouchTips();
        // Vega's own tooltip path: the embed's handler, its formatting and
        // theme, and the nearest point of a line or area under the finger.
        press.view._handler.handleTooltip({clientX: press.x, clientY: press.y}, press.item, true);
        this._placeTouchTip(press.x, press.y);
        this._hoverStart(id, press.row, press.event);
    },
    _touchClose: function(id, event) {
        var t = this._touches[id], tip = t && t.tip;
        if (!tip) return;
        t.tip = null;
        tip.view._handler.handleTooltip({clientX: 0, clientY: 0}, tip.item, false);
        var h = this._hovers[id];
        if (h && h.row === tip.row) this._hoverEnd(id, event);
    },
    _touchReset: function(id) {
        this._touchCancel(id);
        this._touchClose(id, null);
        delete this._touches[id];
    },
    // The part of the page the reader sees, in client coordinates: the
    // visual viewport. On a phone it is smaller than the layout viewport
    // (innerWidth × innerHeight) whenever the page is wider than the screen
    // (a desktop layout, a desktop-view page) or zoomed in.
    _visibleArea: function() {
        var vv = window.visualViewport;
        if (!vv) return {left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight};
        return {left: vv.offsetLeft, top: vv.offsetTop, right: vv.offsetLeft + vv.width, bottom: vv.offsetTop + vv.height};
    },
    // Move the shown tooltip next to client point (x, y), where a finger on
    // it does not cover it: above it, else beside it, else below it, inside
    // the visible area. Vega's tooltip handler puts it below and to the right.
    _placeTouchTip: function(x, y) {
        var tip = document.querySelector('.vg-tooltip.visible');
        if (!tip) return;
        var b = tip.getBoundingClientRect(), w = b.width, h = b.height, g = this._TOUCH_TIP_GAP, v = this._visibleArea();
        var cx = Math.max(v.left, Math.min(v.right - w, x - w / 2)), cy = Math.max(v.top, Math.min(v.bottom - h, y - h / 2));
        var spots = [[cx, y - g - h], [x + g, cy], [x - g - w, cy], [cx, y + 2 * g]], at = [cx, Math.max(v.top, y - g - h)];
        for (var i = 0; i < spots.length; i++) {
            var s = spots[i];
            if (s[0] >= v.left && s[1] >= v.top && s[0] + w <= v.right && s[1] + h <= v.bottom) { at = s; break; }
        }
        tip.style.left = at[0] + 'px';
        tip.style.top = at[1] + 'px';
    },
    // Installed by the first tooltip a tap opens: a press outside its plot
    // (any mouse press) closes it; a scroll, resize, pinch zoom or pan moves
    // it with the tapped spot and closes it once that spot leaves the
    // visible area.
    _watchTouchTips: function() {
        if (this._touchTipListeners) return;
        var self = this;
        var each = function(fn) {
            Object.keys(self._touches).forEach(function(id) {
                var t = self._touches[id], el = self._els[id];
                if (t && t.tip && el) fn(id, t.tip, el);
            });
        };
        var press = function(event) {
            each(function(id, tip, el) {
                if (event.pointerType === 'mouse' || !el.contains(event.target)) self._touchClose(id, event);
            });
        };
        var follow = function(event) {
            var v = self._visibleArea();
            each(function(id, tip, el) {
                var r = el.getBoundingClientRect(), x = r.left + tip.dx, y = r.top + tip.dy;
                if (x < v.left || y < v.top || x > v.right || y > v.bottom) self._touchClose(id, event);
                else self._placeTouchTip(x, y);
            });
        };
        this._touchTipListeners = {press: press, follow: follow};
        document.addEventListener('pointerdown', press, true);
        window.addEventListener('scroll', follow, {capture: true, passive: true});
        window.addEventListener('resize', follow, {passive: true});
        if (window.visualViewport) {
            window.visualViewport.addEventListener('scroll', follow, {passive: true});
            window.visualViewport.addEventListener('resize', follow, {passive: true});
        }
    },

    // --- Keyed highlight ---
    // `AoV.highlight(id, match)` dims every mark of plot `id` whose row does
    // not match to 15 % of its opacity, as a legend selection does; a
    // legend selection and a highlight combine (a mark stays bright only if
    // it passes both). `match` is {field: value} or {field: [values]}
    // (values compare as strings; several fields must all match), or a
    // function datum => boolean. A row that has none of the match's fields
    // (a base map under linked points) is left alone. `AoV.clearHighlight(id)`
    // or `highlight(id, null)` removes it. The highlight is a Vega signal
    // the embedded layers test, so it repaints the live view: zoom, legend
    // selection and the canvas stay. It is kept per plot and re-applied by
    // every re-embed (resize, picker, theme, update_spec).
    _HIGHLIGHT_PARAM: 'aov_highlight',
    _highlightRev: 0,
    highlight: function(id, match) {
        var self = this;
        if (match === null || match === undefined) return self.clearHighlight(id);
        self._highlights[id] = {test: self._highlightTest(match), rev: ++self._highlightRev};
        self._showHighlight(id);
    },
    clearHighlight: function(id) {
        if (!this._highlights[id]) return;
        delete this._highlights[id];
        this._showHighlight(id);
    },
    _highlightTest: function(match) {
        if (typeof match === 'function') return function(d) { return !!match(d); };
        if (typeof match !== 'object') throw new TypeError('AoV.highlight: match is {field: value | [values]}, a function or null; got ' + match);
        var fields = Object.keys(match);
        // A set per field: a link group's hover can name every item of a group.
        var wanted = fields.map(function(f) {
            return new Set((Array.isArray(match[f]) ? match[f] : [match[f]]).map(String));
        });
        // A field the row lacks is not tested, so a row with none of them passes.
        return function(d) {
            for (var i = 0; i < fields.length; i++) {
                if (!d || !Object.prototype.hasOwnProperty.call(d, fields[i])) continue;
                if (!wanted[i].has(String(d[fields[i]]))) return false;
            }
            return true;
        };
    },
    _highlightValue: function(id) {
        var h = this._highlights[id];
        return h ? {id: id, rev: h.rev} : null;
    },
    _showHighlight: function(id) {
        var self = this;
        self.whenReady(id, function(view) {
            try {
                view.signal(self._HIGHLIGHT_PARAM, self._highlightValue(id)).runAsync();
            } catch (e) { self._console.error.call(console, '[' + id + '] AoV.highlight:', e); }
        });
    },
    // The expression function the highlight test calls: whether `datum`
    // stays bright under the highlight the signal value names.
    _registerHighlight: function() {
        if (this._highlightRegistered || typeof vega === 'undefined' || !vega.expressionFunction) return;
        var self = this;
        vega.expressionFunction('aovHighlighted', function(h, datum) {
            var hl = h && self._highlights[h.id];
            return !hl || hl.test(datum);
        });
        this._highlightRegistered = true;
    },
    // The spec a view embeds, with the highlight signal (seeded with the
    // plot's current highlight) and each data layer's dimming condition.
    // A layer without an opacity encoding gets a condition with no else
    // value, so an unhighlighted plot keeps Vega-Lite's own default opacity
    // (0.7 on scatter points); a constant opacity and a legend selection's
    // dimming are extended; a data-driven or other conditional opacity, and
    // composite marks (boxplot, errorbar), are left alone, as the legend
    // selection leaves them.
    _withHighlight: function(id, spec) {
        var self = this, name = self._HIGHLIGHT_PARAM, DIM = 0.15;
        self._registerHighlight();
        spec.params = (spec.params || []).filter(function(p) { return p.name !== name; })
            .concat([{name: name, value: self._highlightValue(id)}]);
        var dimmed = name + ' !== null && !aovHighlighted(' + name + ', datum)';
        self._legendUnits(spec).forEach(function(info) {
            var u = info.unit, enc = info.encoding;
            var mark = typeof u.mark === 'string' ? u.mark : u.mark.type;
            if (['boxplot', 'errorbar', 'errorband'].indexOf(mark) !== -1) return;
            var op = enc.opacity, next;
            if (op === undefined) {
                var base = typeof u.mark === 'object' && typeof u.mark.opacity === 'number' ? u.mark.opacity :
                    ['point', 'tick', 'circle', 'square'].indexOf(mark) !== -1 && !Object.keys(enc).some(function(k) {
                        return enc[k] && typeof enc[k] === 'object' && enc[k].aggregate;
                    }) ? 0.7 : 1;
                next = {condition: {test: dimmed, value: DIM * base}};
            } else if (u._aovLegend && u._aovLegend.dimmed && op.condition && !Array.isArray(op.condition)) {
                next = {condition: {test: {and: [op.condition.test, {not: dimmed}]}, value: op.condition.value},
                    value: op.value};
            } else if (op && typeof op.value === 'number' && op.condition === undefined && op.field === undefined) {
                next = {condition: {test: dimmed, value: DIM * op.value}, value: op.value};
            } else {
                return;
            }
            u.encoding = Object.assign({}, u.encoding, {opacity: next});
        });
        return spec;
    },

    // --- Link groups (`config(link=(; group, key))`) ---
    // Plots and tables showing the same items join a named group. Hovering a
    // linked plot's mark announces its key on `document` as an `htmxo:link`
    // event, detail {group, key, keys, source} (the key as a string, null when
    // the hover ends; `keys` the hovered items' keys, [key] or []; `source` is
    // the plot element), the contract HTMXObjects' tables share. A table's
    // group row (a street of its items) announces all its members' keys, with
    // key null. Every other member of the group shows the keys: a linked plot
    // highlights its marks whose `key` field is one of them, and no keys
    // clears that. An announcement without `keys` shows its `key`. The
    // hovered plot itself is not dimmed.
    _link: function(id) {
        var cur = this._cur[id], link = cur && cur.spec && cur.spec._aov && cur.spec._aov.link;
        return link && link.group ? link : null;
    },
    _announce: function(id, datum) {
        var link = this._link(id);
        if (!link || !datum || !Object.prototype.hasOwnProperty.call(datum, link.key) ||
            datum[link.key] === null || datum[link.key] === undefined) return null;
        var key = String(datum[link.key]);
        this._announceKey(id, key);
        return key;
    },
    _announceKey: function(id, key) {
        var link = this._link(id);
        if (!link) return;
        document.dispatchEvent(new CustomEvent('htmxo:link',
            {detail: {group: link.group, key: key, keys: key === null ? [] : [key], source: this._els[id] || null}}));
    },
    // The keys an announcement shows, as strings: its `keys` list, or the
    // one `key` of a sender that has no list (none when it is null).
    _linkKeys: function(d) {
        var keys = Array.isArray(d.keys) ? d.keys : [d.key];
        return keys.filter(function(k) { return k !== null && k !== undefined; }).map(String);
    },
    // The latest announcement per group, {keys, source}.
    _linkState: {},
    _showLinkKeys: function(id, link, keys) {
        if (!keys.length) { this.clearHighlight(id); return; }
        var m = {};
        m[link.key] = keys;
        this.highlight(id, m);
    },
    _joinLinkState: function(id) {
        var link = this._link(id), st = link && this._linkState[link.group];
        if (st && st.keys.length && st.source !== this._els[id]) this._showLinkKeys(id, link, st.keys);
    },
    _watchLinks: function() {
        if (this._linkListener) return;
        var self = this;
        this._linkListener = function(e) {
            var d = e.detail || {};
            if (typeof d.group !== 'string') return;
            var keys = self._linkKeys(d);
            self._linkState[d.group] = {keys: keys, source: d.source || null};
            Object.keys(self._cur).forEach(function(id) {
                var link = self._link(id);
                if (!link || link.group !== d.group || (d.source && d.source === self._els[id])) return;
                self._showLinkKeys(id, link, keys);
            });
        };
        document.addEventListener('htmxo:link', this._linkListener);
    },

    // --- Plot data download / inline preview ---
    // Read inline data from a plot's view; returns array of objects.
    // Tries the named source first, falls back to common VL conventions.
    _plotData: function(id, name) {
        var view = this.views[id];
        if (!view) return null;
        name = name || 'source_0';
        try { return this._withoutPads(view.data(name)); } catch (e) { /* fall through */ }
        // Fallback: enumerate runtime data, return first non-empty array
        try {
            var runtime = view._runtime && view._runtime.data;
            if (runtime) {
                for (var k in runtime) {
                    try {
                        var d = view.data(k);
                        if (Array.isArray(d) && d.length) return this._withoutPads(d);
                    } catch (e) {}
                }
            }
        } catch (e) {}
        return null;
    },

    // A sparse facet grid's `__aov_pad` rows only keep its cells: they copy a
    // real row's values onto a combination that has no data.
    _withoutPads: function(rows) {
        return rows.filter(function(r) { return !r || r.__aov_pad !== true; });
    },

    // Whether a mark draws values from its rows: it is not a text mark, and a
    // channel that shows values reads a column. Names on a map, an uncoloured
    // region and an outline draw none.
    _NOT_VALUE_CHANNELS: {text: 1, tooltip: 1, description: 1, ariaRoleDescription: 1, href: 1, url: 1, cursor: 1},
    _markDrawsValues: function(mark) {
        if (mark.type === 'text') return false;
        var update = (mark.encode && mark.encode.update) || {}, skip = this._NOT_VALUE_CHANNELS;
        return Object.keys(update).some(function(channel) {
            return !skip[channel] && [].concat(update[channel]).some(function(e) {
                return e && (typeof e.field === 'string' || /\bdatum\b/.test(e.signal || ''));
            });
        });
    },

    // The datasets the marks of a plot draw from, read off the Vega that
    // vega-embed compiled, in the order of its sources:
    // [{name, tags: {<__src value>: drawsValues}, untagged: drawsValues}, ...],
    // where `tags` holds the sources that marks filter the dataset to and
    // `untagged` is for marks that read all of it. A dataset no mark reads is
    // not there, such as the geometry a layer looks up by key. Vega scopes its
    // dataset names by group, so a facet's cells can reuse an outer `data_0`:
    // each name resolves to its root where it is defined.
    _drawnDatasets: function(id) {
        var result = this._results[id], vg = result && result.vgSpec;
        if (!vg) return null;
        var drawn = {}, self = this;
        (function walk(group, outer) {
            var scope = Object.create(outer);
            (group.data || []).forEach(function(d) {
                var up = d.source ? scope[d.source] : {root: d.name};
                if (!up) return;
                var tag = up.tag;
                (d.transform || []).forEach(function(t) {
                    var m = t.type === 'filter' && /datum\.__src === '([^']*)'/.exec(t.expr || '');
                    if (m) tag = m[1];
                });
                scope[d.name] = {root: up.root, tag: tag};
            });
            (group.marks || []).forEach(function(m) {
                var from = m.from || {};
                if (m.type === 'group') {
                    var cells = scope;
                    if (from.facet) {
                        cells = Object.create(scope);
                        cells[from.facet.name] = scope[from.facet.data];
                    }
                    walk(m, cells);
                    return;
                }
                var source = from.data && scope[from.data];
                if (!source) return;
                var d = drawn[source.root] = drawn[source.root] || {name: source.root, tags: {}, untagged: false};
                var values = self._markDrawsValues(m);
                if (source.tag === undefined) d.untagged = d.untagged || values;
                else d.tags[source.tag] = d.tags[source.tag] || values;
            });
        })(vg, {});
        return (vg.data || []).filter(function(d) { return !d.source && drawn[d.name]; })
            .map(function(d) { return drawn[d.name]; });
    },

    // The rows a plot's CSV download and data preview give, [{label, rows}, ...]:
    // one group per dataset that draws values, a merged dataset's sources apart
    // (`_splitBySource`). A figure that draws no values, such as a plain outline
    // map, gives what it draws. With several datasets, those without `__src` are
    // labelled as a merged dataset's would be, `_s1`, `_s2`, … in layer order, so
    // `layer_labels` reads the same on a faceted figure and on one without facets.
    _plotGroups: function(id) {
        var view = this.views[id];
        if (!view) return null;
        var self = this, drawn = this._drawnDatasets(id);
        if (!drawn || !drawn.length) {
            var rows = this._plotData(id);
            return rows && this._splitBySource(rows);
        }
        var nonEmpty = function(g) { return g.rows.length > 0; };
        var all = [], values = [];
        drawn.forEach(function(d, i) {
            self._splitBySource(self._withoutPads(view.data(d.name))).forEach(function(g) {
                var group = {label: g.label, rows: g.rows, place: i + 1};
                all.push(group);
                if (d.untagged || (g.label !== null && d.tags[g.label])) values.push(group);
            });
        });
        var groups = values.filter(nonEmpty);
        if (!groups.length) groups = all.filter(nonEmpty);
        if (!groups.length) return [{label: null, rows: []}];
        return groups.map(function(g) {
            var label = g.label === null && groups.length > 1 ? '_s' + g.place : g.label;
            return {label: label, rows: g.rows};
        });
    },

    // Split rows by a discriminator column (e.g. "__src" for multi-source layered specs).
    // Returns [{label, rows}, ...]. If no rows have the column, returns [{label: null, rows}].
    _splitBySource: function(rows, col) {
        col = col || '__src';
        if (!rows || !rows.length) return [{label: null, rows: rows || []}];
        var hasCol = rows.some(function(r) { return r && Object.prototype.hasOwnProperty.call(r, col); });
        if (!hasCol) return [{label: null, rows: rows}];
        var groups = {};
        var order = [];
        rows.forEach(function(r) {
            var v = r && r[col];
            var k = v === undefined ? '__none__' : String(v);
            if (!groups[k]) { groups[k] = []; order.push(k); }
            // Drop the discriminator from the exported row
            var clean = {};
            for (var f in r) { if (f !== col) clean[f] = r[f]; }
            groups[k].push(clean);
        });
        return order.map(function(k) {
            return {label: k === '__none__' ? null : k, rows: groups[k]};
        });
    },

    _csvEscape: function(s) {
        s = (s === null || s === undefined) ? '' : String(s);
        return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    },

    _rowsToCsv: function(rows) {
        if (!rows || !rows.length) return '';
        var cols = Object.keys(rows[0]);
        var self = this;
        var header = cols.map(self._csvEscape).join(',');
        var body = rows.map(function(r) {
            return cols.map(function(c) { return self._csvEscape(r[c]); }).join(',');
        }).join('\n');
        return header + '\n' + body;
    },

    _triggerDownload: function(text, filename, mime) {
        var blob = new Blob([text], {type: (mime || 'text/csv') + ';charset=utf-8;'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = filename;
        document.body.appendChild(a); a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    // A caption's download menu (`with_plot_caption`), opening or closing. The
    // browser does both; this keeps the open menu at its button while the
    // page scrolls, and closes it when the focus moves on. It sits below the
    // button, or above it when there is little room below and more above, and
    // scrolls within that room. CSS holds it against the button's right edge,
    // or the viewport's left edge, with no need to measure it (`aov.css`).
    _downloadMenuToggle: function(menu, event) {
        if (menu._aovMenuDone) menu._aovMenuDone();
        if (event.newState !== 'open') return;
        var button = menu.previousElementSibling;
        var place = function() {
            if (!menu.isConnected) { menu._aovMenuDone(); return; }
            var r = button.getBoundingClientRect(), root = document.documentElement;
            var below = root.clientHeight - r.bottom, up = below < 160 && r.top > below;
            menu.style.right = (root.clientWidth - r.right) + 'px';
            menu.style.minWidth = r.width + 'px';
            menu.style.top = up ? 'auto' : (r.bottom + 2) + 'px';
            menu.style.bottom = up ? (root.clientHeight - r.top + 2) + 'px' : 'auto';
            menu.style.maxHeight = Math.max(0, (up ? r.top : below) - 6) + 'px';
        };
        var focusout = function(e) {
            var to = e.relatedTarget;
            if (to && !menu.contains(to) && to !== button) menu.hidePopover();
        };
        place();
        window.addEventListener('scroll', place, true);
        window.addEventListener('resize', place);
        menu.addEventListener('focusout', focusout);
        menu._aovMenuDone = function() {
            window.removeEventListener('scroll', place, true);
            window.removeEventListener('resize', place);
            menu.removeEventListener('focusout', focusout);
            menu._aovMenuDone = null;
        };
    },

    // Public: download the plot's data as CSV, one file per group (`_plotGroups`).
    // labels: optional {srcValue: humanLabel} override map.
    downloadPlotData: function(id, filenameBase, labels) {
        filenameBase = filenameBase || id;
        labels = labels || {};
        var groups = this._plotGroups(id);
        if (!groups) { console.warn('AoV.downloadPlotData: no data for', id); return; }
        var self = this;
        groups.forEach(function(g) {
            var label = g.label === null ? '' : '_' + (labels[g.label] || g.label).replace(/[^A-Za-z0-9_-]+/g, '_');
            var fname = filenameBase + label + '.csv';
            self._triggerDownload(self._rowsToCsv(g.rows), fname);
        });
    },

    // Public: download the plot as a PNG/SVG image via vega view.toImageURL.
    // A host-themed plot is transparent on screen; its image is painted
    // on the host's background so its text stays legible as a file.
    downloadPlotImage: function(id, format, filenameBase) {
        filenameBase = filenameBase || id;
        format = (format || 'png').toLowerCase();
        var view = this.views[id];
        if (!view) { console.warn('AoV.downloadPlotImage: no view for', id); return; }
        return this.plotImageURL(id, format).then(function(url) {
            var a = document.createElement('a');
            a.href = url; a.download = filenameBase + '.' + format;
            document.body.appendChild(a); a.click();
            document.body.removeChild(a);
        }).catch(function(err) {
            console.warn('AoV.downloadPlotImage failed:', err);
        });
    },
    // The image URL the PNG/SVG download saves.
    plotImageURL: function(id, format) {
        var view = this.views[id];
        if (!view) return Promise.reject(new Error('AoV.plotImageURL: no view for ' + id));
        if (!(id in this._themeKeys)) return view.toImageURL(format || 'png');
        var prev = view.background();
        view.background(this.hostBackground(this._els[id]));
        var restore = function() { view.background(prev); return view.runAsync(); };
        return view.toImageURL(format || 'png').then(function(url) {
            return restore().then(function() { return url; });
        }, function(err) { return restore().then(function() { throw err; }); });
    },

    // A same-origin vendored <script src> / <link href> (`data-aov-vendor`,
    // emitted by `vega_head(source=:vendor)`) as a promise of the inline
    // <script> / <style> carrying its bytes, for a self-contained page. The
    // copy keeps its marker, so a saved page exports again like an inline one.
    _inlineVendored: function(el) {
        var url = el.src || el.href, isScript = el.tagName === 'SCRIPT';
        var mark = ' data-aov-vendor="' + el.getAttribute('data-aov-vendor').replace(/"/g, '&quot;') + '"';
        return fetch(url).then(function(r) {
            if (!r.ok) throw new Error('AoV.downloadPlotHtml: ' + url + ' answered HTTP ' + r.status);
            return r.text();
        }).then(function(text) {
            // Inline element text must not spell a closing tag that ends the
            // element early or that a text-substituting proxy keys on — the
            // rule of the Julia side's `_inline_text`: `<\/` reads the same as
            // `</` inside JS and CSS strings, and only `</` before those tag
            // names is rewritten (a bare `</` can be a regex's `<` and its `/`).
            // This runtime is itself inlined, so it never spells such a tag.
            var inline = text.replace(/<\/(?=(?:script|style|head|body|html)\b)/gi, '<\\/');
            return isScript ? '<script' + mark + '>' + inline + '<\/script>'
                : '<style' + mark + '>' + inline + '<\/style>';
        });
    },

    // Public: download THIS card as a standalone .html file — no server
    // round-trip. Clones the card's live DOM (picker + plot + caption +
    // lazy shells), resets live-only UI state on the clone, and prepends
    // a <head> collected from the live document (Vega scripts + AoV
    // runtime + caption CSS), mirroring `to_html(::HTMX.Node)`. Returns a
    // promise that settles once the file is handed to the browser.
    downloadPlotHtml: function(id, filenameBase) {
        filenameBase = filenameBase || id;
        var anchor = document.getElementById(id);
        if (!anchor) { console.warn('AoV.downloadPlotHtml: no element for', id); return; }
        var root = anchor.closest('[data-aov-fragment]') || anchor.closest('figure');
        if (!root) { console.warn('AoV.downloadPlotHtml: no card root for', id); return; }
        var clone = root.cloneNode(true);

        // Live-rendered view output: the fresh page re-embeds from the spec.
        var plotEl = clone.querySelector('#' + id);
        if (plotEl) plotEl.innerHTML = '';
        // The plot's font and `--aov-*` look come from page CSS the saved
        // file does not carry: pin them on the saved plot element.
        if (plotEl && id in this._themeKeys) {
            var live = getComputedStyle(anchor);
            this._HOST_LOOK.map(function(p) { return p[0]; }).concat(['--aov-category']).forEach(function(name) {
                var v = live.getPropertyValue(name).trim();
                if (v) plotEl.style.setProperty(name, v);
            });
            var font = JSON.parse(this._themeKeys[id]).font;
            if (font) plotEl.style.setProperty('--aov-font', font);
        }
        // Frame players and ribbon switches: the fresh page's embed builds
        // its own controls.
        clone.querySelectorAll('[data-aov-frame-for], [data-aov-ribbon-view-for], .aov-ribbon-view').forEach(function(c) { c.remove(); });
        // A download menu's place is set whenever it opens.
        clone.querySelectorAll('.aov-download-menu').forEach(function(m) { m.removeAttribute('style'); });
        // Lazy data shells: drop rendered bodies; the fresh page re-renders lazily.
        clone.querySelectorAll('.aov-data-raw-body[data-aov-plot-id], .aov-data-pretty-body[data-aov-plot-id]').forEach(function(b) {
            b.innerHTML = '';
        });
        clone.querySelectorAll('[data-aov-rendered]').forEach(function(el) {
            el.removeAttribute('data-aov-rendered');
        });
        // Pretty/Raw toggle: back to the initial Pretty mode.
        clone.querySelectorAll('details.aov-data-preview[data-mode]').forEach(function(d) {
            d.dataset.mode = 'pretty';
            d.querySelectorAll('button[data-view]').forEach(function(btn) {
                if (btn.dataset.view === 'pretty') btn.setAttribute('aria-pressed', 'true');
                else btn.removeAttribute('aria-pressed');
            });
        });
        // Picker: restore the INITIAL pin/disabled/checked state. Pin changes
        // (and URL restore) flip select.disabled, a reflecting IDL attribute,
        // so the clone's attributes may disagree with the authored initial
        // state — read the initial pin from the picker's own inline script.
        var pin0 = null;
        clone.querySelectorAll('script').forEach(function(s) {
            var m = /_aovPin_\w+_current = '([A-Za-z_]+)'/.exec(s.textContent || '');
            if (m) pin0 = m[1];
        });
        if (pin0) {
            clone.querySelectorAll('select[id^="aov-remap-"][id$="-' + id + '"]').forEach(function(sel) {
                var ch = sel.id.slice('aov-remap-'.length, sel.id.length - id.length - 1);
                var radio = clone.querySelector('input[name="aov-pin-' + id + '"][value="' + ch + '"]');
                // Fixed channels keep their authored disabled select (their radio
                // is disabled too and _aovPin never touches radios); x/y/off radios
                // are disabled without disabling their selects.
                var fixedCh = radio && radio.disabled && ch !== 'x' && ch !== 'y' && ch !== 'off';
                if (ch === pin0 || fixedCh) sel.setAttribute('disabled', 'disabled');
                else sel.removeAttribute('disabled');
            });
            clone.querySelectorAll('input[name="aov-pin-' + id + '"]').forEach(function(r) {
                if (r.value === pin0) r.setAttribute('checked', 'checked');
                else r.removeAttribute('checked');
            });
        }

        // <head> pieces, in document order: the same Vega scripts, runtime,
        // page settings and caption CSS/JS the live page runs. CDN tags stay
        // tags; vendored files (`data-aov-vendor`) end up inline — fetched
        // when loaded by URL — so the saved file needs neither this server
        // nor a CDN.
        var self = this;
        var headParts = [];
        document.querySelectorAll('script, style, link[data-aov-vendor]').forEach(function(s) {
            if (s.hasAttribute('data-aov-vendor')) {
                // Vendored by URL: fetch the bytes; already inline: copy as is.
                headParts.push(s.hasAttribute('src') || s.hasAttribute('href') ? self._inlineVendored(s) : s.outerHTML);
            } else if (s.tagName === 'SCRIPT' && s.hasAttribute('src')) {
                if (/\/vega(-lite|-embed)?@/.test(s.getAttribute('src') || '')) headParts.push(s.outerHTML);
            } else if (s.tagName === 'SCRIPT') {
                var t = s.textContent || '';
                if (t.indexOf('window.AoV = window.AoV ||') !== -1 ||
                    t.indexOf('window.AoV = Object.assign(window.AoV') !== -1 ||
                    t.indexOf('function sortTable(') !== -1) {
                    headParts.push(s.outerHTML);
                }
            } else {
                var c = s.textContent || '';
                if (c.indexOf('aov-data-preview') !== -1 || c.indexOf('caption-actions') !== -1) {
                    headParts.push(s.outerHTML);
                }
            }
        });

        // The data URLs the plot loaded, written into the card after the runtime.
        headParts.push(self._preloadScript(id));

        var text = this._texts(anchor);
        var title = text('untitled');
        var capTitle = clone.querySelector('figcaption .caption-header strong');
        if (capTitle && capTitle.textContent) title = capTitle.textContent;
        else if (document.title) title = document.title;
        title = title.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

        return Promise.all(headParts).then(function(parts) {
            var lang = text('lang').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
            var page = '<!DOCTYPE html>\n<html lang="' + lang + '">\n<head>\n<meta charset="utf-8">\n' +
                '<title>' + title + '<\/title>\n' +
                '<style>body{font-family:system-ui,sans-serif;margin:1rem}<\/style>\n' +
                parts.join('\n') + '\n<\/head>\n<body>\n' +
                clone.outerHTML + '\n<\/body>\n<\/html>\n';
            self._triggerDownload(page, filenameBase + '.html', 'text/html');
        });
    },

    // The words AoV writes into a plot's chrome, as `text(name, values)`. A
    // captioned plot's figure carries non-English texts as `data-aov-texts`
    // (`plot_texts`, src/plot_texts.jl); without that attribute they are
    // English. LOCKSTEP: `_TEXTS_EN` is the English set of
    // `_RUNTIME_TEXT_NAMES` in src/plot_texts.jl; test/plot_texts.jl pins them.
    _TEXTS_EN: {"no_data": "(no data available)", "no_summary": "(no summary data available)", "median": "Median", "interval": "{point} [{width} credible interval]", "value": "value", "untitled": "AoV plot", "lang": "en", "play": "Play", "pause": "Pause", "ribbons": "Ribbons", "intervals": "Intervals", "ribbon_view": "Drawing", "overlap": "Bands overlap {score} of the y range", "overlap_auto": "Bands overlap {score} of the y range; intervals above {threshold}", "scrollable": "Scrollable plot", "scrollable_table": "Scrollable table"},
    _texts: function(el) {
        var host = el && el.closest ? el.closest('[data-aov-texts]') : null;
        var texts = host ? JSON.parse(host.dataset.aovTexts) : {};
        var en = this._TEXTS_EN;
        return function(name, values) {
            var t = texts[name] !== undefined ? texts[name] : en[name];
            return t.replace(/\{([A-Za-z_]+)\}/g, function(m, k) {
                return values && values[k] !== undefined ? String(values[k]) : m;
            });
        };
    },

    // Public: render the plot's data as sortable HTML table(s) into `container`.
    // Builds lazily — call from the <details> "toggle" event.
    showPlotData: function(id, container, labels) {
        labels = labels || {};
        if (container.dataset.aovRendered === '1') return;
        var groups = this._plotGroups(id);
        if (!groups) { container.textContent = this._texts(container)('no_data'); container.dataset.aovRendered = '1'; return; }
        container.innerHTML = '';
        var self = this;
        groups.forEach(function(g) {
            if (g.label !== null) {
                var h = document.createElement('h6');
                h.textContent = labels[g.label] || g.label;
                h.style.margin = '0.5rem 0 0.25rem';
                container.appendChild(h);
            }
            container.appendChild(self._buildSortableTable(g.rows));
        });
        this._frameTables(container);
        container.dataset.aovRendered = '1';
    },

    // A table wider or longer than its place scrolls in a frame of its own
    // (aov.css), so the page keeps its width and whatever sits beside the
    // figure stays where it is. Each table has its own frame, so a heading above
    // it stays in view. A table that is already in a frame (a saved page) is
    // only watched. The frame takes the focus while it scrolls, because a
    // table's cells do not (see _syncScrollFrame).
    _frameTables: function(root) {
        var self = this;
        root.querySelectorAll('table').forEach(function(table) {
            var frame = table.closest('.aov-table-frame');
            if (!frame) {
                frame = document.createElement('div');
                frame.className = 'aov-table-frame';
                table.parentNode.insertBefore(frame, table);
                frame.appendChild(table);
            }
            self._watchTableFrame(frame, table);
        });
    },
    // A container that changes width, or a table that does (a late font, a
    // shown pane), starts or stops the scrolling. One observer serves every
    // frame, and a frame that leaves the page leaves it too.
    _watchTableFrame: function(frame, table) {
        var self = this;
        if (window.ResizeObserver) {
            if (!self._tableFrameObserver) {
                self._tableFrameObserver = new ResizeObserver(function(entries) {
                    entries.forEach(function(entry) {
                        var watched = entry.target.closest('.aov-table-frame');
                        if (watched) self._syncTableFrame(watched);
                    });
                });
            }
            self._tableFrameObserver.observe(frame);
            self._tableFrameObserver.observe(table);
        }
        self._syncTableFrame(frame);
    },
    _syncTableFrame: function(frame) {
        var self = this;
        this._syncFrameStop(frame, function() {
            var word = self._texts(frame)('scrollable_table'), title = self._captionTitle(frame);
            return title ? word + ': ' + title : word;
        });
    },

    _buildSortableTable: function(rows, cols) {
        var table = document.createElement('table');
        table.className = 'striped';
        table.setAttribute('role', 'grid');
        if (!rows || !rows.length) {
            table.innerHTML = '<tbody><tr><td>(empty)</td></tr></tbody>';
            return table;
        }
        if (!cols) {
            var rawCols = Object.keys(rows[0]);
            var stringCols = [], numericCols = [];
            rawCols.forEach(function(c) {
                var isNumeric = false;
                for (var i = 0; i < rows.length; i++) {
                    var v = rows[i][c];
                    if (v === null || v === undefined || v === '') continue;
                    isNumeric = (typeof v === 'number');
                    break;
                }
                (isNumeric ? numericCols : stringCols).push(c);
            });
            stringCols.sort();
            numericCols.sort();
            cols = stringCols.concat(numericCols);
        }
        var thead = document.createElement('thead');
        var trh = document.createElement('tr');
        cols.forEach(function(c, i) {
            var th = document.createElement('th');
            th.textContent = c;
            th.style.cursor = 'pointer';
            th.onclick = function() {
                if (typeof window.sortTable === 'function') window.sortTable(i, th);
            };
            trh.appendChild(th);
        });
        thead.appendChild(trh);
        table.appendChild(thead);
        var tbody = document.createElement('tbody');
        rows.forEach(function(r) {
            var tr = document.createElement('tr');
            cols.forEach(function(c) {
                var td = document.createElement('td');
                var v = r[c];
                td.textContent = (v === null || v === undefined) ? '' : String(v);
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        return table;
    },

    // Public: toggle between Pretty and Raw views inside a captioned plot's
    // <details>. Sets details[data-mode] (CSS keys off it to show/hide
    // [data-view] wrappers) and aria-pressed on the toggle buttons.
    // Lazily renders the chosen view on first switch.
    toggleDataView: function(btn, view) {
        var details = btn.closest('details');
        if (!details) return;
        details.dataset.mode = view;
        var group = details.querySelector('[role="group"]');
        if (group) {
            group.querySelectorAll('button[data-view]').forEach(function(b) {
                if (b.dataset.view === view) b.setAttribute('aria-pressed', 'true');
                else b.removeAttribute('aria-pressed');
            });
        }
        this._lazyRenderDataView(details, view);
    },

    _lazyRenderDataView: function(details, view) {
        if (view === 'raw') {
            var body = details.querySelector('.aov-data-raw-body');
            if (body && body.dataset.aovRendered !== '1') {
                var pid = body.dataset.aovPlotId;
                var labels = body.dataset.aovLabels ? JSON.parse(body.dataset.aovLabels) : {};
                if (pid) this.showPlotData(pid, body, labels);
            }
        } else if (view === 'pretty') {
            var body = details.querySelector('.aov-data-pretty-body');
            if (body && body.dataset.aovRendered !== '1') {
                var pid = body.dataset.aovPlotId;
                var opts = body.dataset.aovSummaryOpts ? JSON.parse(body.dataset.aovSummaryOpts) : {};
                if (pid) this.renderPrettySummary(pid, body, opts);
            }
        }
        // Also a table the caller wrote (`summary_table=`).
        this._frameTables(details);
    },

    // Public: render a "point [lo, hi]" pretty summary table from the live
    // Vega view's source_0, which already holds the aggregated columns
    // produced by Julia. Two modes:
    //   1. Auto-detect: looks for __point__ or __median__ as the central col,
    //      and lo_<prob>_ / hi_<prob>_ pairs for the intervals (used by
    //      pointinterval/gradient/dot/lineribbon).
    //   2. Explicit (opts.bands): caller passes [[loCol, hiCol, label], ...]
    //      and opts.point_col (used by lineribbon(bands=...) precomputed).
    //
    //   opts.ci: 'outer' (default, widest band) | 'inner' | Number (closest prob)
    //   opts.sigfigs: significant figures for numeric formatting (default 2)
    //   opts.value_label: header for the formatted value column (default 'value')
    //   opts.point_label: 'Median' (default) | 'Mean' | custom — used in caption
    renderPrettySummary: function(id, container, opts) {
        opts = opts || {};
        if (container.dataset.aovRendered === '1') return;
        var text = this._texts(container);
        var groups = this._summaryGroups(id, opts);
        if (!groups || !groups.some(function(g) { return g.rows.length; })) {
            container.textContent = text('no_summary');
            container.dataset.aovRendered = '1';
            return;
        }
        container.innerHTML = '';
        var labels = opts.labels || {};
        var self = this;
        groups.forEach(function(g) {
            if (g.label !== null) {
                var heading = document.createElement('h6');
                heading.textContent = labels[g.label] || g.label;
                heading.style.margin = '0.5rem 0 0.25rem';
                container.appendChild(heading);
            }
            var built = self._buildPrettyRows(g.rows, opts, text);
            if (!built) {
                container.appendChild(self._buildSortableTable(g.rows));
                return;
            }
            if (built.caption) {
                var cap = document.createElement('figcaption');
                cap.textContent = built.caption;
                cap.style.cssText = 'font-size:0.85em;opacity:0.75;margin:0 0 0.25rem';
                container.appendChild(cap);
            }
            container.appendChild(self._buildSortableTable(built.rows, built.cols));
        });
        this._frameTables(container);
        container.dataset.aovRendered = '1';
    },

    // The sources of a plot's first dataset that holds a summary. Another layer's
    // rows come first when that layer does (observations under a ribbon), each
    // in a dataset of its own. Without a summary, the plot's data as usual.
    _summaryGroups: function(id, opts) {
        var view = this.views[id];
        if (!view) return null;
        var self = this;
        var hasSummary = function(g) { return self._buildPrettyRows(g.rows, opts) !== null; };
        for (var i = 0; ; i++) {
            var rows;
            try { rows = this._withoutPads(view.data('source_' + i)); } catch (e) { break; }
            var groups = this._splitBySource(rows);
            if (groups.some(hasSummary)) return groups;
        }
        return this._plotGroups(id);
    },

    // LOCKSTEP: `plot_summary_table` (src/plot_summary_table.jl) is this
    // computation for a lowered spec; test/plot_summary_table.jl compares them.
    _buildPrettyRows: function(rows, opts, text) {
        if (!rows || !rows.length) return null;
        text = text || this._texts(null);
        var first = rows[0];
        var pointCol = opts.point_col ||
            ('__point__' in first ? '__point__' :
             ('__median__' in first ? '__median__' : null));
        if (!pointCol || !(pointCol in first)) return null;

        var bands; // [{lo, hi, label}, ...] sorted inner→outer
        if (opts.bands && opts.bands.length) {
            // AoV passes explicit bands outermost-first (lineribbon convention);
            // internal representation is inner→outer to match the auto-detect path.
            bands = opts.bands.slice().reverse().map(function(b) {
                return {lo: b[0], hi: b[1], label: b[2] || (b[0] + ' / ' + b[1])};
            });
        } else {
            var probMap = {};
            Object.keys(first).forEach(function(c) {
                var m = /^lo_(\d+(?:_\d+)?)_$/.exec(c);
                if (m && ('hi_' + m[1] + '_') in first) {
                    probMap[m[1]] = parseFloat(m[1].replace('_', '.'));
                }
            });
            var probKeys = Object.keys(probMap);
            if (!probKeys.length) return null;
            probKeys.sort(function(a, b) { return probMap[a] - probMap[b]; });
            bands = probKeys.map(function(k) {
                return {
                    lo: 'lo_' + k + '_',
                    hi: 'hi_' + k + '_',
                    label: Math.round(probMap[k] * 100) + '%',
                    prob: probMap[k]
                };
            });
        }

        var pick;
        var ci = opts.ci;
        if (typeof ci === 'number' && bands[0].prob !== undefined) {
            pick = bands[0]; var bestDiff = Math.abs(pick.prob - ci);
            bands.forEach(function(b) {
                var d = Math.abs(b.prob - ci);
                if (d < bestDiff) { pick = b; bestDiff = d; }
            });
        } else if (ci === 'inner') {
            pick = bands[0];
        } else {
            pick = bands[bands.length - 1];
        }

        var sigfigs = (opts.sigfigs === undefined) ? 2 : opts.sigfigs;
        var fmt = function(x) {
            if (x === null || x === undefined || (typeof x === 'number' && isNaN(x))) return '';
            if (typeof x !== 'number') return String(x);
            return parseFloat(x.toPrecision(sigfigs)).toString();
        };
        var label = opts.value_label || text('value');
        var pointLabel = opts.point_label || text('median');

        var hidden = {};
        hidden[pointCol] = 1;
        bands.forEach(function(b) { hidden[b.lo] = 1; hidden[b.hi] = 1; });

        var visibleSourceCols = Object.keys(first).filter(function(c) {
            if (hidden[c]) return false;
            if (c.indexOf('__') === 0) return false;
            return true;
        });
        // Categorical first (alpha), then numeric (alpha), then the value col.
        var stringCols = [], numericCols = [];
        visibleSourceCols.forEach(function(c) {
            var isNumeric = false;
            for (var i = 0; i < rows.length; i++) {
                var v = rows[i][c];
                if (v === null || v === undefined || v === '') continue;
                isNumeric = (typeof v === 'number');
                break;
            }
            (isNumeric ? numericCols : stringCols).push(c);
        });
        stringCols.sort(); numericCols.sort();
        var keyCols = stringCols.concat(numericCols);
        var orderedCols = keyCols.concat([label]);

        var roundNum = function(v) {
            if (typeof v !== 'number' || isNaN(v)) return v;
            return parseFloat(v.toPrecision(sigfigs));
        };
        // Julia's summaries come in hash order, so read along the key columns.
        var empty = function(v) { return v === null || v === undefined; };
        var built = rows.slice().sort(function(a, b) {
            for (var i = 0; i < keyCols.length; i++) {
                var x = a[keyCols[i]], y = b[keyCols[i]];
                if (empty(x) || empty(y)) {
                    if (empty(x) && empty(y)) continue;
                    return empty(x) ? 1 : -1;
                }
                if (typeof x !== 'number' || typeof y !== 'number') { x = String(x); y = String(y); }
                if (x !== y) return x < y ? -1 : 1;
            }
            return 0;
        }).map(function(r) {
            var out = {};
            visibleSourceCols.forEach(function(c) { out[c] = roundNum(r[c]); });
            out[label] = fmt(r[pointCol]) + ' [' + fmt(r[pick.lo]) + ', ' + fmt(r[pick.hi]) + ']';
            return out;
        });

        var caption = pick.prob !== undefined ?
            text('interval', {point: pointLabel, width: pick.label}) :
            pointLabel + ' [' + pick.label + ']';
        return {rows: built, cols: orderedCols, caption: caption};
    },

    // Wire a signal to an HTMX GET request.
    // Standalone (no-server) pages load no HTMX: degrade silently instead
    // of throwing a ReferenceError from the debounced callback.
    signalToHtmx: function(id, signal, url, target, swap, debounceMs) {
        debounceMs = debounceMs || 300;
        var timer = null;
        this.onSignal(id, signal, function(name, value) {
            clearTimeout(timer);
            timer = setTimeout(function() {
                if (typeof htmx === 'undefined' || !htmx.ajax) return;
                var params = typeof value === 'object' ? value : {};
                var qs = Object.keys(params).map(function(k) {
                    return encodeURIComponent(k) + '=' + encodeURIComponent(JSON.stringify(params[k]));
                }).join('&');
                var fullUrl = qs ? url + '?' + qs : url;
                htmx.ajax('GET', fullUrl, {target: target, swap: swap || 'innerHTML'});
            }, debounceMs);
        });
    },

    // Client-side encoding remapping: swap color/row/column fields without server round-trip
    // Re-facet a plot client-side (the channel picker). The mapping is kept
    // (minus its data copy) so update_spec can re-apply it to a refreshed
    // spec when no picker builder is registered for the plot.
    remapEncoding: function(id, mapping) {
        var spec = this._remappedSpec(id, mapping);
        if (!spec) return;
        // Ribbon groups that rows sent since the first render brought.
        var live = (this._liveRows[id] || {}).source_0;
        if (live) spec = this._withRibbonGroups(spec, live) || spec;
        var kept = Object.assign({}, mapping);
        delete kept._comboData;
        this._mappings[id] = kept;
        // Re-embed, but preserve the TRUE original spec
        var savedOrig = this._origSpecs[id];
        this._embed(id, spec, this._embedOpts[id], true);
        this._origSpecs[id] = savedOrig;
    },

    // The colour encoding a remap assigns to `field`. A field the layer
    // was authored with keeps its authored scale (palette, category
    // order), sort, legend and conditions (`lowclip`/`highclip`); any
    // other field gets the default scheme.
    _remappedColor: function(authored, field, title) {
        var enc = {field: field, type: 'nominal', title: title};
        if (authored && authored.field === field) {
            ['scale', 'sort', 'legend', 'condition'].forEach(function(k) {
                if (authored[k] !== undefined) enc[k] = authored[k];
            });
        }
        return enc;
    },

    // The `calculate` expression of a ribbon interval drawing's dodge on
    // the ribbon's own x axis: `field` moved by the step of the row's
    // `by` group in `domain`, added on a linear axis, multiplied in as
    // `exp` of a log step on a log one. The same formula as
    // `_ribbon_dodge_expr` (src/ribbon_view.jl); a row outside `domain`
    // stays at its x.
    _dodgeXExpr: function(field, by, domain, span, scale) {
        var f = 'datum[' + JSON.stringify(field) + ']';
        var n = domain.length;
        if (!by || n < 2) return 'isValid(' + f + ') ? ' + f + ' : null';
        var k = 'indexof(' + JSON.stringify(domain) + ', datum[' + JSON.stringify(by) + '])';
        var d = '(' + k + ' < 0 ? 0 : (' + k + ' - ' + (n - 1) / 2 + ') * ' + span / n + ')';
        return 'isValid(' + f + ') ? ' + (scale === 'log' ? f + ' * exp(' + d + ')' : f + ' + ' + d) + ' : null';
    },

    // Whether a layer was given a fixed colour (`visual(...; color=...)`):
    // a literal mark colour, or AoV's theme-following ink style
    // (`color=:ink`), which a composite mark (errorbar, boxplot) carries
    // on its drawn parts. A colour remap leaves such layers alone, as the
    // server-side picker does (`_visual_static_channels`).
    _fixedMarkColor: function(l) {
        var mark = l && l.mark;
        if (!mark || typeof mark !== 'object') return false;
        var ink = function(m) {
            var style = m && typeof m === 'object' ? m.style : undefined;
            return style === 'aov-ink' || (Array.isArray(style) && style.indexOf('aov-ink') !== -1);
        };
        return !!mark.color || ink(mark) ||
            Object.keys(mark).some(function(k) { return ink(mark[k]); });
    },

    // The spec `remapEncoding` embeds: the stored original with `mapping`
    // applied. Embeds nothing.
    _remappedSpec: function(id, mapping) {
        var self = this;
        var orig = this._origSpecs[id];
        if (!orig) { console.warn('AoV.remapEncoding: no stored spec for', id); return null; }
        var spec = JSON.parse(JSON.stringify(orig));

        // If combo data was pre-built by the caller (multi-select picker),
        // inject it into the cloned spec so combos are available to all channels
        if (mapping._comboData && mapping._comboData.values) {
            if (spec.data && spec.data.values) spec.data = mapping._comboData;
            else if (spec.spec && spec.spec.data) spec.spec.data = mapping._comboData;
        }

        // Per-column-Y concat (`hconcat` of per-column facet views —
        // `scales(Y=(; scale=Dict(...)))`): remap color/row/detail/axes in
        // EVERY child, preserving each child's Y scale type and column
        // filter. Column remapping is not supported — the per-column scale
        // keys bind to the authored column field — so it logs and no-ops
        // (the picker should pin the column channel on these figures).
        if (spec.hconcat && Array.isArray(spec.hconcat)) {
            // Local title lookup — the shared `_titles`/`_fieldTitle` pair is
            // declared further down, so at this point the hoisted `var` is
            // still undefined (proven live: `_titles[f]` TypeError).
            var _titlesC = mapping._comboTitles || {};
            function _fieldTitleC(f) { return _titlesC[f] || f; }
            if ('column' in mapping) {
                console.info('AoV.remapEncoding: column remap is not supported on per-column-scale (hconcat) figures; keeping the authored column field.');
            }
            if ('color' in mapping) {
                var cfC = mapping.color;
                spec.hconcat.forEach(function(child) {
                    if (!child || !child.spec || !Array.isArray(child.spec.layer)) return;
                    child.spec.layer.forEach(function(l) {
                        if (!l || !l.encoding || l._keep_color || self._fixedMarkColor(l)) return;
                        if (cfC) {
                            l.encoding.color = self._remappedColor(l.encoding.color, cfC, _fieldTitleC(cfC));
                        } else {
                            delete l.encoding.color;
                        }
                    });
                });
            }
            if ('row' in mapping) {
                var rfC = mapping.row;
                spec.hconcat.forEach(function(child) {
                    if (!child || !child.facet) return;
                    if (rfC) {
                        child.facet.row = {field: rfC, type: 'nominal', title: _fieldTitleC(rfC)};
                    } else {
                        delete child.facet.row;
                    }
                });
            }
            if (mapping._dimensions) {
                var detailFieldsC = mapping._dimensions;
                spec.hconcat.forEach(function(child) {
                    if (!child || !child.spec || !Array.isArray(child.spec.layer)) return;
                    child.spec.layer.forEach(function(l) {
                        if (!l || !l.encoding) return;
                        if (detailFieldsC.length > 0) {
                            l.encoding.detail = detailFieldsC.length === 1 ?
                                {field: detailFieldsC[0], type: 'nominal'} :
                                detailFieldsC.map(function(f) { return {field: f, type: 'nominal'}; });
                        } else {
                            delete l.encoding.detail;
                        }
                    });
                });
            }
            function _remapConcatAxis(axis) {
                if (!(axis in mapping)) return;
                var newField = mapping[axis];
                if (!newField) return;
                var valsC = spec.data && Array.isArray(spec.data.values) ? spec.data.values : null;
                var sampleC = null;
                if (valsC) {
                    for (var i = 0; i < valsC.length; i++) {
                        if (valsC[i] && valsC[i][newField] !== undefined && valsC[i][newField] !== null) { sampleC = valsC[i][newField]; break; }
                    }
                }
                var newTypeC = sampleC !== null ? _inferType(sampleC) : null;
                spec.hconcat.forEach(function(child) {
                    if (!child || !child.spec || !Array.isArray(child.spec.layer)) return;
                    child.spec.layer.forEach(function(l) {
                        if (!l || l._no_axis_remap || !l.encoding) return;
                        var encC = l.encoding[axis];
                        if (!encC || typeof encC !== 'object' || encC.field === undefined) return;
                        var oldField = encC.field;
                        encC.field = newField;
                        if (newTypeC) {
                            encC.type = newTypeC;
                            if (newTypeC !== 'quantitative' && encC.scale && encC.scale.type) {
                                var stC = encC.scale.type;
                                if (stC === 'log' || stC === 'sqrt' || stC === 'pow') delete encC.scale.type;
                            }
                        }
                        encC.title = _fieldTitleC(newField);
                        if (Array.isArray(encC.tooltip)) {
                            encC.tooltip.forEach(function(t) {
                                if (t && t.field === oldField) {
                                    t.field = newField;
                                    t.title = _fieldTitleC(newField);
                                    if (newTypeC) t.type = newTypeC;
                                }
                            });
                        }
                    });
                });
            }
            _remapConcatAxis('x');
            _remapConcatAxis('y');

            // Re-broadcast cross-source layers after row mutations
            this._broadcastCrossSource(spec);

            return spec;
        }

        // Find layers in either simple or faceted structure
        var isFaceted = !!(spec.facet || (spec.spec && spec.spec.layer));
        var layers = isFaceted ? (spec.spec && spec.spec.layer || []) : (spec.layer || [spec]);

        // Migrate encoding-based row/column to facet key structure
        // (encoding.row/column is VL inline faceting that conflicts with facet key)
        function _migrateEncodingFacets() {
            var enc = spec.encoding || (spec.spec && spec.spec.encoding);
            if (!enc) {
                // Check sublayer encodings
                layers.forEach(function(l) {
                    if (!l.encoding) return;
                    ['row', 'column'].forEach(function(ch) {
                        if (l.encoding[ch]) {
                            if (!isFaceted) { wrapFaceted(); }
                            spec.facet = spec.facet || {};
                            spec.facet[ch] = spec.facet[ch] || l.encoding[ch];
                            delete l.encoding[ch];
                        }
                    });
                });
                return;
            }
            ['row', 'column'].forEach(function(ch) {
                if (enc[ch]) {
                    if (!isFaceted) { wrapFaceted(); }
                    spec.facet = spec.facet || {};
                    spec.facet[ch] = spec.facet[ch] || enc[ch];
                    delete enc[ch];
                }
            });
        }
        _migrateEncodingFacets();

        // Resolve combo titles for human-readable legend/facet headers
        var _titles = mapping._comboTitles || {};
        function _fieldTitle(f) { return _titles[f] || f; }

        // Color remapping
        if ('color' in mapping) {
            var cf = mapping.color;
            var lrMeta = orig._aov && orig._aov.lineribbon;
            if (lrMeta && lrMeta.templateLayers) {
                // Lineribbon per-group layering: rebuild layers from template
                var tmpl = lrMeta.templateLayers;
                var newLayers = [];
                // When the spec uses merged data (with __src filters added by
                // layers_to_vl), the rebuilt LR layers must inherit the same
                // __src filter so they don't render rows from other sources.
                var srcFilter = null;
                for (var li = 0; li < layers.length; li++) {
                    var lll = layers[li];
                    if (lll && lll._lr_layer && Array.isArray(lll.transform)) {
                        for (var ti = 0; ti < lll.transform.length; ti++) {
                            var tf = lll.transform[ti];
                            if (tf && typeof tf.filter === 'string' && tf.filter.indexOf('__src') !== -1) {
                                srcFilter = tf;
                                break;
                            }
                        }
                        if (srcFilter) break;
                    }
                }
                if (cf) {
                    // One group's layers per value of the new color field in the data
                    var proto = tmpl.map(function(tl) {
                        var gl = JSON.parse(JSON.stringify(tl));
                        gl.transform = srcFilter ? [srcFilter] : [];
                        gl.encoding.color = {field: cf, type: 'nominal', title: _fieldTitle(cf)};
                        return gl;
                    });
                    var groups = self._rowGroups(spec.data && spec.data.values || [], cf);
                    // No group yet: the template layers stand in (as the server emits them).
                    newLayers = groups.length ? self._ribbonGroupLayers(proto, cf, groups) :
                        proto.map(function(gl) { gl._lr_layer = true; gl._lr_proto = true; return gl; });
                } else {
                    // No color: use template layers as-is (with __src filter if any)
                    tmpl.forEach(function(tl) {
                        var gl = JSON.parse(JSON.stringify(tl));
                        if (srcFilter) gl.transform = [srcFilter];
                        gl._lr_layer = true;
                        newLayers.push(gl);
                    });
                }
                // Preserve non-lineribbon layers (e.g. cross-source dose VLines)
                // and replace only the tagged lineribbon layers.
                var preserved = layers.filter(function(l) { return !(l && l._lr_layer); });
                var allLayers = preserved.concat(newLayers);
                if (isFaceted) {
                    spec.spec.layer = allLayers;
                } else {
                    spec.layer = allLayers;
                }
                layers = allLayers;
            } else {
                // Non-lineribbon: standard color remapping. Skip layers
                // given a fixed colour (a literal or `:ink`, `_fixedMarkColor`)
                // — e.g. black observation scatters or a theme-ink ribbon —
                // which should not be data-driven by the picker.
                // An interval dodged by its colour (the auto-dodge, and a
                // ribbon's interval drawing) keeps dodging by the colour:
                // left on the old field, a field moved to a facet would
                // draw every panel's interval off centre.
                var dodged = null;
                layers.forEach(function(l) {
                    var c = l && l.encoding && l.encoding.color;
                    if (dodged === null && c && typeof c.field === 'string' && !l._keep_color &&
                            !self._fixedMarkColor(l)) dodged = c.field;
                });
                layers.forEach(function(l) {
                    if (!l || !l.encoding || dodged === null || dodged === cf) return;
                    ['xOffset', 'yOffset'].forEach(function(ch) {
                        var off = l.encoding[ch];
                        if (!off || off.field !== dodged) return;
                        if (cf) l.encoding[ch] = {field: cf, type: 'nominal'}; else delete l.encoding[ch];
                    });
                });
                // A ribbon's interval drawing on the ribbon's own x axis
                // dodges with a `calculate` (`_aov.dodgeX`,
                // src/ribbon_view.jl): rebuilt for the new colour's groups,
                // or undone without a colour.
                var dx = spec._aov && spec._aov.dodgeX;
                if (dx && dodged !== null && dodged === dx.by && dodged !== cf) {
                    var dxRows = (spec.data && spec.data.values) || [];
                    var dxDomain = null;
                    layers.forEach(function(l) {
                        var ts = l && Array.isArray(l.transform) ? l.transform : [];
                        ts.forEach(function(t) {
                            if (!t || t.as !== dx.as) return;
                            if (dxDomain === null) {
                                var tag = null;
                                ts.forEach(function(u) {
                                    var m = u && typeof u.filter === 'string' &&
                                        /^datum\.__src === '([^']*)'$/.exec(u.filter);
                                    if (m) tag = m[1];
                                });
                                dxDomain = cf ? self._rowGroups(dxRows.filter(function(r) {
                                    return r && r.__aov_pad !== true && (tag === null || r.__src === tag);
                                }), cf) : [];
                            }
                            t.calculate = self._dodgeXExpr(dx.field, cf, dxDomain, dx.span, dx.scale);
                        });
                    });
                    spec._aov.dodgeX = Object.assign({}, dx, {by: cf || null, domain: dxDomain || []});
                }
                layers.forEach(function(l) {
                    if (!l.encoding) return;
                    if (l._keep_color) return;  // layer-fixed color (field outside remappable dims)
                    if (self._fixedMarkColor(l)) return;
                    if (cf) {
                        l.encoding.color = self._remappedColor(l.encoding.color, cf, _fieldTitle(cf));
                    } else {
                        delete l.encoding.color;
                    }
                    // Sync tooltips: remove old color entries, add new
                    if (Array.isArray(l.encoding.tooltip)) {
                        l.encoding.tooltip = l.encoding.tooltip.filter(function(t) {
                            return t.type !== 'nominal' || t.field === (spec.facet && spec.facet.row && spec.facet.row.field) ||
                                   t.field === (spec.facet && spec.facet.column && spec.facet.column.field);
                        });
                        if (cf) l.encoding.tooltip.push({field: cf, type: 'nominal'});
                    }
                });
            }
        }

        // Count unique values for a field in the data (for nFacetCols hint)
        function countUnique(field) {
            return Math.max(self._distinctCount(spec.data && spec.data.values, field), 1);
        }

        // Unlinked facet axes (`config(facet=(; linkyaxes=:none))`) belong to
        // the facet: beside a flat `layer`, an independent x/y scale gives every
        // layer its own scale and axis. Julia keeps them off a flat spec and
        // records them as `_aovFacetResolve` (`_merge_facet_resolve!`), so they
        // come with the facet a remap adds and go with the one it removes.
        function facetResolve(faceted) {
            var scale = spec.resolve && spec.resolve.scale;
            if (faceted) {
                var unlinked = spec._aovFacetResolve && spec._aovFacetResolve.scale;
                if (!unlinked) return;
                spec.resolve = Object.assign({}, spec.resolve,
                    {scale: Object.assign({}, scale, unlinked)});
                return;
            }
            if (!scale) return;
            ['x', 'y'].forEach(function(ax) {
                if (scale[ax] === 'independent') delete scale[ax];
            });
            if (!Object.keys(scale).length) delete spec.resolve.scale;
            if (!Object.keys(spec.resolve).length) delete spec.resolve;
        }

        // Helper: wrap a non-faceted spec into faceted structure
        function wrapFaceted() {
            if (isFaceted) return;
            // A container-sized plot stays container-sized in facets; one with an
            // authored width keeps it as its cell width.
            var responsive = !!((spec._aov && spec._aov.fitWidth) || spec.width === 'container');
            if (spec.layer) {
                spec.spec = {layer: spec.layer};
                delete spec.layer;
            } else {
                // Single-view spec: move mark+encoding into spec.spec
                var inner = {};
                ['mark', 'encoding', 'transform', 'selection', 'params', '_aovLegend'].forEach(function(k) {
                    if (spec[k] !== undefined) { inner[k] = spec[k]; delete spec[k]; }
                });
                // Selections belong to the unit; variable parameters (a
                // frame's aov_frame) stay on the top level, where every
                // panel reads them.
                if (inner.params) {
                    var vars = inner.params.filter(function(p) { return !p.select; });
                    if (vars.length) {
                        spec.params = vars;
                        inner.params = inner.params.filter(function(p) { return p.select; });
                        if (!inner.params.length) delete inner.params;
                    }
                }
                spec.spec = inner;
            }
            // Width/height describe the child view in a facet operator.
            // Leaving height on the outer object loses the authored size;
            // a categorical axis then falls back to step sizing (and VL's
            // nearest-point overlay can compile an unbound datum expression).
            ['width', 'height'].forEach(function(k) {
                if (spec[k] !== undefined) { spec.spec[k] = spec[k]; delete spec[k]; }
            });
            spec.facet = {};
            facetResolve(true);
            // Remove single-view-only properties from outer spec
            delete spec.autosize;
            // Add _aov hint for responsive faceted sizing
            spec._aov = spec._aov || {};
            if (responsive) spec._aov.fitWidth = true;
            isFaceted = true;
            layers = spec.spec.layer || [spec.spec];
        }

        // Helper: unwrap faceted spec back to flat
        function unwrapFaceted() {
            if (!spec.facet) return;
            if (spec.facet.column || spec.facet.row) return;
            var inner = spec.spec || {};
            if (inner.layer) {
                spec.layer = inner.layer;
                ['width', 'height'].forEach(function(k) {
                    if (inner[k] !== undefined) spec[k] = inner[k];
                });
            } else {
                // Restore single-view keys
                Object.keys(inner).forEach(function(k) {
                    spec[k] = k === 'params' && spec.params ? spec.params.concat(inner[k]) : inner[k];
                });
            }
            delete spec.spec;
            delete spec.facet;
            facetResolve(false);
            if (!spec.layer) {
                if (spec._aov && spec._aov.fitWidth) {
                    // Single-view: use VL native responsive width
                    spec.width = 'container';
                    spec.autosize = {type: 'fit', contains: 'padding'};
                    var kept = {};
                    ['frame', 'link'].forEach(function(k) { if (spec._aov[k]) kept[k] = spec._aov[k]; });
                    delete spec._aov;
                    if (Object.keys(kept).length) spec._aov = kept;
                }
                // An authored width came back from the cell above.
            } else {
                // Layered: use _aov marker for JS responsive sizing
                // (preserve keys such as maxWidth).
                spec._aov = spec._aov || {};
            }
            isFaceted = false;
            layers = spec.layer || [spec];
        }

        // Row facet remapping
        if ('row' in mapping) {
            var rf = mapping.row;
            if (rf) {
                wrapFaceted();
                spec.facet.row = {field: rf, type: 'nominal', title: _fieldTitle(rf)};
            } else if (spec.facet) {
                delete spec.facet.row;
                unwrapFaceted();
            }
        }

        // Column facet remapping
        if ('column' in mapping) {
            var clf = mapping.column;
            if (clf) {
                wrapFaceted();
                spec.facet.column = {field: clf, type: 'nominal', title: _fieldTitle(clf)};
            } else if (spec.facet) {
                delete spec.facet.column;
                unwrapFaceted();
            }
        }

        // Update _aov.nFacetCols for responsive sizing
        if (isFaceted && spec._aov && spec.facet) {
            if (spec.facet.column) {
                spec._aov.nFacetCols = countUnique(spec.facet.column.field);
            } else if (spec.facet.field) {
                // A wrap facet lays out at most `columns` panels per row.
                var nWrap = countUnique(spec.facet.field);
                spec._aov.nFacetCols = spec.columns ? Math.min(nWrap, spec.columns) : nWrap;
            } else {
                delete spec._aov.nFacetCols;
            }
        }

        // Detail encoding: put explicitly-specified dimension fields into detail
        // so VL groups by them without assigning visual properties.
        // With the multi-select picker, _dimensions is the resolved detail
        // channel contents (no set subtraction needed).
        if (mapping._dimensions) {
            var detailFields = mapping._dimensions;
            layers.forEach(function(l) {
                if (!l.encoding) return;
                if (detailFields.length > 0) {
                    l.encoding.detail = detailFields.length === 1 ?
                        {field: detailFields[0], type: 'nominal'} :
                        detailFields.map(function(f) { return {field: f, type: 'nominal'}; });
                } else {
                    delete l.encoding.detail;
                }
            });
        }

        // x/y axis remapping. Each swap: rewrite encoding.<axis>.field, re-infer
        // the type from a sample value in the data, refresh axis title and any
        // tooltip entry referencing the old field. Layers tagged `_no_axis_remap`
        // (planned: analysis layers with value axes bound to computed columns)
        // are skipped.
        function _inferType(v) {
            if (typeof v === 'number') return 'quantitative';
            if (typeof v === 'string' && /^\d{4}-\d{2}-\d{2}/.test(v)) return 'temporal';
            return 'nominal';
        }
        function _sampleVals() {
            if (spec.data && Array.isArray(spec.data.values)) return spec.data.values;
            if (spec.spec && spec.spec.data && Array.isArray(spec.spec.data.values)) return spec.spec.data.values;
            return null;
        }
        function _remapAxis(axis) {
            if (!(axis in mapping)) return;
            var newField = mapping[axis];
            if (!newField) return;  // empty = don't change (defensive: avoids breaking mandatory axes)
            var vals = _sampleVals();
            var sample = null;
            if (vals) {
                for (var i = 0; i < vals.length; i++) {
                    if (vals[i] && vals[i][newField] !== undefined && vals[i][newField] !== null) {
                        sample = vals[i][newField]; break;
                    }
                }
            }
            var newType = sample !== null ? _inferType(sample) : null;
            layers.forEach(function(l) {
                if (!l || l._no_axis_remap) return;
                var enc = l.encoding;
                if (!enc || !enc[axis] || typeof enc[axis] !== 'object') return;
                var oldField = enc[axis].field;
                if (oldField === undefined) return;
                enc[axis].field = newField;
                if (newType) {
                    enc[axis].type = newType;
                    // When the axis type flips from quantitative to nominal, a log/sqrt
                    // scale no longer makes sense — drop it rather than leave VL with
                    // an invalid scale it will ignore with warnings.
                    if (newType !== 'quantitative' && enc[axis].scale && enc[axis].scale.type) {
                        var st = enc[axis].scale.type;
                        if (st === 'log' || st === 'sqrt' || st === 'pow') {
                            delete enc[axis].scale.type;
                        }
                    }
                }
                enc[axis].title = _fieldTitle(newField);
                var tt = enc.tooltip;
                if (Array.isArray(tt)) {
                    tt.forEach(function(t) {
                        if (t && t.field === oldField) {
                            t.field = newField;
                            t.title = _fieldTitle(newField);
                            if (newType) t.type = newType;
                        }
                    });
                }
            });
        }
        _remapAxis('x');
        _remapAxis('y');

        // Re-broadcast cross-source layers after row/column mutations
        // (color mutations need no broadcast — see _broadcastCrossSource).
        this._broadcastCrossSource(spec);

        return spec;
    }
};
