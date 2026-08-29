VISTA — paper figures, individual files
=======================================
All PNGs are 600 dpi. Figures marked "1:1" were built at their intended print
size, so include them at that width and DO NOT rescale — the point sizes are
already correct and scaling will undo that.

IN THE PAPER RIGHT NOW
----------------------
TDL.pdf                      Fig 1  teaser schematic. Vector, cropped out of
                                    final_submission.pdf. width=\columnwidth
fig_spider_all.png           Fig 2  overlaid radar, all 9 series.
                                    1:1 at \columnwidth (3.031in). 
fig_flip_rate_bar.png        Fig 3  overall flip-rate scoreboard.
                                    NOT rebuilt — still an 8x5in canvas, so it
                                    shrinks ~0.38x at \columnwidth. Ask if you
                                    want it rebuilt 1:1 like the others.
fig_heatmaps_4models.png     Fig 4  2x2 heatmap (Gemma, Qwen, Haiku, GPT-5-mini).
                                    1:1 at \textwidth (6.3in) in a figure*.
fig_modifier_type_all.png    Fig 5  modifier-type grouped bar.
                                    NOT rebuilt — 11x5.2in canvas, shrinks
                                    ~0.28x at \columnwidth. This is the one you
                                    flagged as too small; it still needs the
                                    1:1 rebuild.
per_scenario_flip_rates.png  Fig 6  per-scenario flip rates (appendix).
                                    NOT rebuilt.

HEATMAPS AS SEPARATE PANELS
---------------------------
heatmap_gemma.png     heatmap_qwen.png      heatmap_llama8b.png
heatmap_haiku.png     heatmap_gpt41mini.png heatmap_gpt5mini.png
heatmap_sonnet.png

One model each, all seven. Every panel shares ONE colour scale (vmax = 43.2%,
the global max across models), so any subset you place together stays
comparable. Each is 1:1 at \columnwidth. The blue box marks that panel's
strongest cell — SC004_2 x self_preservation in Gemma, Qwen, Haiku, GPT-5-mini.

Note: the ORIGINAL heatmaps (heatmap_scenario_axis*.png, from
tools/make_paper_figures.py) set vmax per panel, so Gemma's 42.1% and Qwen's
31.6% were both painted as the darkest cell. Don't mix those with these.

ALTERNATES (not currently used)
-------------------------------
fig_spider_small_multiples.png  radar faceted one panel per system, 1:1 at
                                \textwidth. Fixes the overlap in Fig 2 at the
                                cost of the single-shape read.
fig_axis_heatmap.png            systems x axes heatmap with values printed.
                                Most legible form; fits one column.

REGENERATING
------------
make_fig_spider_zoom.py      -> fig_spider_all.png
make_fig_spider_v2.py        -> fig_spider_small_multiples.png, fig_axis_heatmap.png
make_fig_heatmaps4.py        -> fig_heatmaps_4models.png
make_heatmaps_separate.py    -> heatmap_<model>.png
make_figures_extended.py     -> fig_flip_rate_bar.png, fig_modifier_type_all.png
