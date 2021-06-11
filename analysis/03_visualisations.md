Extra visualisations of the mapping between ACT-R and LBA
================
Maarten van der Velde
Last updated: 2021-06-11

## Background

In ACT-R, the retrieval time of a chunk with activation \(A_i\) is

\[RT = F * e^{-A_i} + t_{er}\]

In this equation, \(F\) is a “latency factor”, and \(t_{er}\) comprises
all non-retrieval processes (stimulus encoding, response preparation and
execution).

ACT-R retrieval is equivalent to an LBA, which becomes apparent when we
rewrite the equation to this form:

\[RT = \frac{F}{e^{A_i} } + t_{er}\]

Response time in an LBA is defined as follows:

\[RT = \frac{b - A/2}{v} + t_0\]

Here, \(b\) is the response boundary, \(A\) the upper bound of the
uniform starting point distribution (the average starting point is
\(A/2\)), \(v\) the drift rate, and \(t_0\) the non-decision time.

In both models, the choice between response alternatives is determined
by a race process: ACT-R selects the option with the largest activation
(which by definition also has the smallest RT); LBA selects the option
with the smallest RT (which by definition also has the largest drift
rate).

We can make a direct mapping of ACT-R parameters onto LBA parameters:

| **ACT-R**  | **LBA**     |
| ---------- | ----------- |
| \(F\)      | \(b - A/2\) |
| \(A_i\)    | \(ln(v)\)   |
| \(t_{er}\) | \(t_0\)     |

# Setup

``` r
library(dplyr)
library(ggplot2)
library(rtdists)
library(tidyr)
library(cowplot)
library(grid)
library(rlang)

theme_paper <- theme_classic(base_size = 14) + 
  theme(axis.text = element_text(colour = "black"))


set.seed(2021)
```

# ACT-R as an LBA

Visualise the ACT-R model in the style of an accumulator model.

Number of trials to simulate:

``` r
n_trials <- 1e5
```

Set the ACT-R parameters:

``` r
# Latency factor F
lf <- 2

# Non-retrieval time t_er
t_er <- 1

# Activation of correct answer
a_c_mu <- -.5
a_c_sd <- 1

# Activation of incorrect answer
a_f_mu <- -1.5
a_f_sd <- 1.5
```

``` r
sim_actr_viz <- tibble(
  f = rep(lf, n_trials),
  a_c = rnorm(n_trials, mean = a_c_mu, sd = a_c_sd),
  a_f = rnorm(n_trials, mean = a_f_mu, sd = a_f_sd),
  t = rep(t_er, n_trials)
) %>%
  rowwise() %>%
  mutate(rt = f * exp(-max(a_c, a_f)) + t,
         response = ifelse(a_c > a_f, 1, 2)) %>%
  ungroup()

sim_actr_sample <- sample_n(sim_actr_viz, 150)
```

``` r
p_actr_lba <- ggplot() +
  
  # t_er rectangle
  geom_rect(aes(xmax = t_er, xmin = 0, ymin = 0, ymax = 1), fill = "grey80") +
  geom_segment(aes(xend = t_er, x = 0, y = .5, yend = .5), lty = 1,
               arrow = arrow(ends = "both", type = "closed", length = unit(2, "mm"))) +
  geom_text(aes(x = t_er/2, y = .7, label = paste0(expression(t[er]))), parse = TRUE, size = rel(5)) +
  
  # Top density plot
  geom_density(data = sim_actr_viz,
               aes(x = rt, y = after_stat(count)/(nrow(sim_actr_viz)/2) + lf + .5, colour = as.factor(response)), 
               n = 2^10,
               adjust = .05) +
  
  # Dotted y-axis
  geom_vline(xintercept = 0, lty = 3) +
  
  # F arrow
  geom_segment(aes(x = 0, xend = 0, y = .5, yend = lf + .5),
               arrow = arrow(ends = "both", type = "closed", length = unit(2, "mm"))) +
  geom_text(aes(y = lf/2 + .5, x = .2, label = "F"), size = rel(5)) +
  
  # Example trajectories
  geom_point(data = sim_actr_sample, aes(x = rt, y = f + .5, colour = as.factor(response)), alpha = .25) +
  geom_segment(data = sim_actr_sample, aes(x = t, xend = rt, y = .5, yend = f + .5, colour = as.factor(response)), alpha = .1, lwd = rel(.5)) +

  # Mean drift rate lines
  geom_segment(aes(x = t_er, xend = lf/exp(a_c_mu) + t_er, yend = lf + .5), y = .5, colour = "#0571b0", lty = 1) +
  geom_path(aes(x = (lf/exp(a_c_mu)) / 2 + t_er + c(0, .4, .4, 0), 
                y = lf/2 + .5 + c(0, 0, exp(a_c_mu)*.4, 0))) +
  geom_label(aes(x = (lf/exp(a_c_mu)) / 2 + t_er + .8, y = lf/2 + .5 + exp(a_c_mu)*.2, 
                 label = paste0(expression(e^mu[c]))),
             parse = TRUE, label.size = NA, size = rel(5), label.padding = unit(.1, "lines")) +
  
  geom_segment(aes(x = t_er, xend = lf/exp(a_f_mu) + t_er, yend = lf + .5), y = .5, colour = "#ca0020", lty = 1) +
  geom_path(aes(x = (lf/exp(a_f_mu)) / 2 + t_er + c(0, .4, .4, 0), 
                y = lf/2 + .5 + c(0, 0, exp(a_f_mu)*.4, 0))) +
  geom_label(aes(x = (lf/exp(a_f_mu)) / 2 + t_er + .8, y = lf/2 + .5 + exp(a_f_mu)*.2, 
                label = paste0(expression(e^mu[f]))),
             parse = TRUE, label.size = NA, size = rel(5), label.padding = unit(.1, "lines")) +

  
  # Boundary line
  geom_hline(yintercept = lf + .5) +

  scale_x_continuous(expand = c(.0075,0), breaks = NULL) +
  coord_cartesian(xlim = c(0, 12), clip = "off") +
  scale_y_continuous(expand = c(0,0), limits = c(0, NA),
                     breaks = c(0, .5, 1, lf + .5),
                     labels = c(0, "A/2", "A", "d")) +
  scale_colour_manual(values = c("#0571b0", "#ca0020")) +
  labs(x = "Time",
       y = NULL) +
  guides(colour = FALSE) +
  theme_paper +
  theme(axis.ticks.y = element_blank(),
        axis.line.y = element_blank(),
        axis.text.y = element_text(colour = "grey50"))

p_actr_lba
```

![](03_visualisations_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "sim-actr-wide.pdf"), width = 9, height = 3)
```

# Comparison between models

Verify that with the same set of parameters, ACT-R and the LBA produce
the same RT distribution.

``` r
sim_actr <- tibble(
  f = rep(lf, n_trials),
  a_c = rnorm(n_trials, mean = a_c_mu, sd = a_c_sd),
  a_f = rnorm(n_trials, mean = a_f_mu, sd = a_f_sd),
  t = rep(t_er, n_trials)
) %>%
  rowwise() %>%
  mutate(rt = f * exp(-max(a_c, a_f)) + t,
         response = ifelse(a_c > a_f, 1, 2)) %>%
  ungroup() %>%
  mutate(model = "ACT-R",
         rt = ifelse(response == 1, rt, -rt))

# F = b - A/2 -> b = F + A/2
# Set A to 1 to find the value of b
A <- 1
b <- lf + .5*A

dlba_dat <- expand.grid(rt = seq(0, 20, by = .01),
                        response = c(1, 2))

sim_lba <- bind_cols(dlba_dat,
                     density = dLBA(rt = dlba_dat$rt,
                                    response = dlba_dat$response,
                                    A = A,
                                    b = b,
                                    t0 = t_er,
                                    meanlog_v = c(a_c_mu, a_f_mu),
                                    sdlog_v = c(a_c_sd, a_f_sd),
                                    distribution = "lnorm",
                                    silent = TRUE)) %>%
  mutate(rt = ifelse(response == 1, rt, -rt),
         model = "LBA")
```

``` r
draw_key_custom <- function(data, params, size) {
  if(data$colour == "#000000" && data$size == .5) { # ACT-R
    grobTree(
      linesGrob(
        c(.1, .1, .3, .3, .3, .5, .5, .5, .7, .7, .7, .9, .9),
        c(0, .5, .5, 0, .8, .8, 0, .65, .65, 0, .4, .4, 0)
      ),
      gp = gpar(
        col = data$colour %||% "grey20",
        fill = alpha(data$fill %||% "white", data$alpha),
        lwd = (data$size %||% 0.5) * .pt,
        lty = data$linetype %||% 1
      )
    )
  } 
  else if (data$colour == "#e66101" && data$size == 2) { # LBA
    grobTree(
      linesGrob(
        c(0, 1),
        c(.5, .5)
      ),
      gp = gpar(
        col = alpha(data$colour %||% "grey20", data$alpha),
        fill = alpha(data$fill %||% "white", data$alpha),
        lwd = (data$size %||% 0.5) * .pt,
        lty = data$linetype %||% 1
      )
    )
  }
  else {
    grobTree() # Don't draw
  }
}

p_rt_comp <- ggplot(sim_actr, aes(x = rt, colour = model)) +
  geom_vline(xintercept = 0, lty = 2, colour = "grey80") +
  geom_histogram(aes(y = ..density..), binwidth = .5, fill = NA, size = rel(.5), key_glyph = draw_key_custom) +
  geom_line(data = sim_lba, aes(y = density), size = rel(2), alpha = .7, key_glyph = draw_key_custom) +
  scale_x_continuous(limits = c(-20, 20)) +
  scale_y_continuous(limits = c(0, max(sim_lba$density) + .01), expand = c(0, 0)) +
  scale_colour_manual(values = c("#000000", "#e66101")) +
  labs(x = "RT (s)",
       y = "Density",
       colour = NULL) +
  theme_paper +
  theme(axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position = c(.85, .85))

p_rt_comp
```

    ## Warning: Removed 1227 rows containing non-finite values (stat_bin).

    ## Warning: Removed 2 rows containing missing values.

![](03_visualisations_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Combination plot for in the paper:

``` r
plot_grid(p_actr_lba, p_rt_comp,
          labels = "AUTO",
          align = "h",
          axis = "tb",
          rel_widths = c(1, .7))
```

    ## Warning: Removed 1227 rows containing non-finite values (stat_bin).

    ## Warning: Removed 2 rows containing missing values.

![](03_visualisations_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "sim-actr-combi.pdf"), width = 9, height = 3)
```

Make separate PNGs for in the
presentation:

``` r
ggsave(plot = p_actr_lba, filename = file.path("..", "output", "sim-actr-presentation.png"), width = 4.5, height = 3, dpi = 600)

ggsave(plot = p_rt_comp, filename = file.path("..", "output", "sim-rt-comparison-presentation.png"), width = 4.5, height = 3, dpi = 600)
```

    ## Warning: Removed 1227 rows containing non-finite values (stat_bin).

    ## Warning: Removed 2 rows containing missing values.

# Session info

``` r
sessionInfo()
```

    ## R version 3.6.3 (2020-02-29)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: Ubuntu 18.04.5 LTS
    ## 
    ## Matrix products: default
    ## BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.7.1
    ## LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=nl_NL.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=nl_NL.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=nl_NL.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=nl_NL.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ## [1] grid      stats     graphics  grDevices utils     datasets  methods  
    ## [8] base     
    ## 
    ## other attached packages:
    ## [1] rlang_0.4.10   cowplot_0.9.4  tidyr_1.0.0    rtdists_0.11-2
    ## [5] ggplot2_3.3.2  dplyr_0.8.3   
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] Rcpp_1.0.6        pillar_1.4.2      compiler_3.6.3   
    ##  [4] tools_3.6.3       digest_0.6.19     evd_2.3-3        
    ##  [7] jsonlite_1.6      lifecycle_0.1.0   evaluate_0.14    
    ## [10] tibble_2.1.3      gtable_0.3.0      lattice_0.20-41  
    ## [13] pkgconfig_2.0.2   Matrix_1.2-18     yaml_2.2.0       
    ## [16] mvtnorm_1.1-1     expm_0.999-4      xfun_0.21        
    ## [19] withr_2.3.0       stringr_1.4.0     knitr_1.23       
    ## [22] vctrs_0.3.8       tidyselect_1.1.1  glue_1.3.1       
    ## [25] R6_2.4.0          survival_2.44-1.1 rmarkdown_2.6    
    ## [28] purrr_0.3.2       magrittr_1.5      scales_1.0.0     
    ## [31] htmltools_0.3.6   splines_3.6.3     assertthat_0.2.1 
    ## [34] colorspace_1.4-1  labeling_0.3      stringi_1.4.3    
    ## [37] gsl_2.1-6         munsell_0.5.0     msm_1.6.8        
    ## [40] crayon_1.3.4
