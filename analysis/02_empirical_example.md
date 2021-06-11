Example Application: Modelling Changing Retrieval Performance in
Empirical Data
================
Maarten van der Velde
Last updated: 2021-06-11

# Overview

In this analysis, we model changes in retrieval performance between
three retrieval practice sessions. First, we analyse behavioural
measures (RT and accuracy). Then. we fit the LBA to the data, convert
the resulting LBA parameters to ACT-R parameters, and analyse those
parameter estimates.

# Setup

``` r
library(dplyr)
library(ggplot2)
library(rtdists)
library(purrr)
library(furrr)
library(tidyr)
library(truncdist)
library(cowplot)
library(grid)
library(rlang)
library(lme4)
library(lmerTest)

future::plan("multiprocess", workers = 6) # Set to desired number of cores

theme_paper <- theme_classic(base_size = 14) + 
  theme(axis.text = element_text(colour = "black"))


use_cached_results <- TRUE # Set to FALSE to rerun simulations (takes time!)

set.seed(2021)
```

## Parameter recovery functions

Define the objective function to minimise (adapted to lognormal
distribution from the [rtdists
example](https://github.com/rtdists/rtdists/blob/master/examples/examples.lba.R)):

``` r
obj_fun <- function(par, rt, response, distribution = "lnorm") {
  # simple parameters
  spar <- par[!grepl("[12]$", names(par))]  
  
  # distribution parameters:
  dist_par_names <- unique(sub("[12]$", "", grep("[12]$" ,names(par), value = TRUE)))
  dist_par <- vector("list", length = length(dist_par_names))
  names(dist_par) <- dist_par_names
  for (i in dist_par_names) dist_par[[i]] <- as.list(unname(par[grep(i, names(par))]))
  dist_par$sdlog_v <- c(1, dist_par$sdlog_v) # fix first sdlog_v to 1

  # get summed log-likelihood:
  d <- do.call(dLBA, args = c(rt=list(rt), response=list(response), spar, dist_par, 
                               distribution=distribution, silent=TRUE))
  if (any(d < 0e-10)) return(1e6) 
  else return(-sum(log(d)))
}
```

Define the parameter recovery function, which randomly initialises the
LBA parameters (within some reasonable constraints to promote
convergence) and then uses the nlminb optimiser to find the best fit,
evaluating each iteration with the objective function:

``` r
recover_parameters <- function(data, obj_fun) {
  
  # Generate random starting values
  init_par <- runif(6)
  init_par[2] <- init_par[2] + 1 # Ensure b is larger than A
  init_par[3] <- runif(1, 0, min(data$rt)) # Ensure t0 is mot too large
  init_par[4] <- -init_par[4] # Ensure meanlog_v1 is negative
  init_par[5] <- init_par[4] - init_par[5] # Ensure meanlog_v2 is negative and lower than meanlog_v2
  names(init_par) <- c("A", "b", "t0", "meanlog_v1", "meanlog_v2", "sdlog_v2")
  
  # Run optimiser
  fit <- nlminb(obj_fun,
                start = init_par, 
                rt = data$rt, response = data$response,
                lower = c(0, 0, 0, -Inf, -Inf, 0)) # Set lower bounds on parameters
  
  # Only keep parameter estimates if the optimiser converged successfully
  if (fit$convergence == 0 && !is.infinite(fit$objective)) {
    return(as.list(c(fit$par, objective = fit$objective)))
  }
  
  return(NULL)
  
}
```

## Prepare data

We’ll clean the data by removing study trials, trials in which no
response was recorded, and trials with an RT lower than 300 ms. To give
the model a chance of fitting the RT distributions, we’ll also require
that participants completed at least 50 trials and made at least 5
errors in each of the three sessions.

``` r
d <- read.csv(file.path("..", "data", "data.csv")) %>%
  filter(study == FALSE,
         !is.infinite(rt),
         rt >= 300) %>%
  transmute(participant = as.character(subject),
            list = case_when( # Simplify list names
              list == "2019-09_1" ~ 1,
              list == "2019-09_2" ~ 2,
              list == "2019-09_3" ~ 3,
              list == "2021-03_1" ~ 1,
              list == "2021-03_2" ~ 2,
              list == "2021-03_3" ~ 3
            ),
            trial_index,
            rt = rt / 1000,
            response = ifelse(correct == TRUE, 1, 2)
  )

# Total number of participants
length(unique(d$participant))
```

    ## [1] 127

``` r
# Full dataset size
nrow(d)
```

    ## [1] 29441

``` r
min_trials <- 50
min_errors <- 5

d <- d %>%
  group_by(participant, list) %>%
  filter(n() > min_trials) %>%
  filter(length(response[response == 2]) > min_errors) %>%
  group_by(participant) %>%
  filter(length(unique(list)) == 3)

# Number of included participants
length(unique(d$participant))
```

    ## [1] 50

``` r
# Number of included trials
nrow(d)
```

    ## [1] 12568

Number of trials per participant, per list:

``` r
d %>%
  group_by(participant, list) %>%
  tally() %>%
  pull(n) %>%
  summary()
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   53.00   75.00   83.00   83.79   91.00  139.00

# Behavioural results

Plot accuracy and RT on correct responses:

``` r
p_acc <- d %>%
  mutate(correct = response == 1) %>%
  group_by(participant, list) %>%
  summarise(accuracy = mean(correct)) %>%
  mutate(list_jitter = jitter(list, .4)) %>%
  ggplot(aes(x = list_jitter, y = accuracy, group = participant)) +
  geom_line(colour = "#78B7C5", alpha = .25) +
  geom_point(colour = "#3B9AB2", alpha = .8, size = .8) +
  geom_boxplot(aes(x = list, group = list), width = .2, colour = "black", outlier.shape = NA, fill = NA) +
  scale_x_continuous(breaks = c(1, 2, 3)) +
  scale_y_continuous(limits = c(.4, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Session",
       y = NULL,
       title = "Accuracy") +
  guides(colour = FALSE) +
  theme_paper +
  theme(plot.title = element_text(hjust = .5, size = rel(1)))

p_acc
```

![](02_empirical_example_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
p_rt <- d %>%
  filter(response == 1) %>%
  group_by(participant, list) %>%
  summarise(rt = median(rt)) %>%
  mutate(list_jitter = jitter(list, .4)) %>%
  ggplot(aes(x = list_jitter, y = rt, group = participant)) +
  geom_line(colour = "#78B7C5", alpha = .25) +
  geom_point(colour = "#3B9AB2", alpha = .8, size = .8) +
  geom_boxplot(aes(x = list, group = list), width = .2, colour = "black", outlier.shape = NA, fill = NA) +
  scale_x_continuous(breaks = c(1, 2, 3)) +
  scale_y_continuous(limits = c(1.75, 5)) +
  labs(x = "Session",
       y = NULL,
       title = "RT (s)") +
  guides(colour = FALSE) +
  theme_paper +
  theme(plot.title = element_text(hjust = .5, size = rel(1)))

p_rt
```

![](02_empirical_example_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
plot_grid(p_acc, p_rt,
          align = "hv",
          axis = "tblr",
          labels = "AUTO")
```

![](02_empirical_example_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "real_data.pdf"), width = 4.5, height = 3)
```

Presentation version:

``` r
plot_grid(p_acc, p_rt,
          align = "hv",
          axis = "tblr")
```

![](02_empirical_example_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "real_data.png"), width = 4.5, height = 3, dpi = 600)
```

Do accuracy and RT change from session to session?

``` r
m_acc <- glmer(correct ~ session + (1 | participant),
               data = mutate(d, 
                             correct = response == 1,
                             session = factor(list, levels = c(2, 1, 3))),
               family = binomial)

summary(m_acc)
```

    ## Generalized linear mixed model fit by maximum likelihood (Laplace
    ##   Approximation) [glmerMod]
    ##  Family: binomial  ( logit )
    ## Formula: correct ~ session + (1 | participant)
    ##    Data: 
    ## mutate(d, correct = response == 1, session = factor(list, levels = c(2,  
    ##     1, 3)))
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##  11668.7  11698.5  -5830.4  11660.7    12564 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.7770  0.3435  0.4161  0.4876  0.9334 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.2181   0.467   
    ## Number of obs: 12568, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  1.63542    0.07854  20.822  < 2e-16 ***
    ## session1    -0.26991    0.05767  -4.680 2.87e-06 ***
    ## session3    -0.01479    0.05851  -0.253      0.8    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.391       
    ## session3 -0.385  0.525

Yes, accuracy increases from session 1 to session 2, but there’s no
evidence of change from session 2 to session 3.

``` r
m_rt <- glmer(rt ~ session + (1 | participant),
              data = mutate(d,
                            session = factor(list, levels = c(2, 1, 3))),
              family = Gamma(link = "identity"))

summary(m_rt)
```

    ## Generalized linear mixed model fit by maximum likelihood (Laplace
    ##   Approximation) [glmerMod]
    ##  Family: Gamma  ( identity )
    ## Formula: rt ~ session + (1 | participant)
    ##    Data: mutate(d, session = factor(list, levels = c(2, 1, 3)))
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##  50100.6  50137.8 -25045.3  50090.6    12563 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -1.4493 -0.6308 -0.2696  0.3244 10.8493 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.1481   0.3849  
    ##  Residual                0.4049   0.6363  
    ## Number of obs: 12568, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error t value Pr(>|z|)    
    ## (Intercept)  3.87471    0.08326  46.538  < 2e-16 ***
    ## session1     0.11610    0.04420   2.627  0.00862 ** 
    ## session3    -0.31292    0.04075  -7.679 1.61e-14 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.254       
    ## session3 -0.273  0.519

RT decreases from session 1 to session 2, and from session 2 to session
3.

# Infer parameters from data

``` r
lists <- unique(d$list)
participants <- unique(d$participant)

if (use_cached_results) {
  
  param_infer_best <- readRDS(file.path("..", "output", "param_infer_best.rds"))
  
} else {
  
  n_attempts <- 250
  param_infer <- list()
  
  for(i in 1:length(participants)) {
    for(j in 1:length(lists)) {
      param_infer <- append(param_infer,
                            list(future_map_dfr(1:n_attempts, function(x) {
                              recover_parameters(data = filter(d,
                                                               participant == participants[i],
                                                               list == lists[j]),
                                                 obj_fun = obj_fun)
                            }) %>%
                              mutate(participant = participants[i],
                                     list = lists[j])
                            )
      )
    }
  }
  
  param_infer <- bind_rows(param_infer)
  
  saveRDS(param_infer, file.path("..", "output", "param_infer.rds"))
  
  # Only keep the best fit across all attempts, which is closest to the global optimum:
  param_infer_best <- param_infer %>%
    group_by(participant, list) %>%
    filter(objective == min(objective)) %>%
    slice(n()) %>% # If there are multiple rows with the exact same value, only keep the first
    ungroup()

  saveRDS(param_infer_best, file.path("..", "output", "param_infer_best.rds"))
  
  
}
```

``` r
dlba_dat <- crossing(participant = participants,
                     list = lists,
                     rt = seq(0, 20, by = .1),
                     response = c(1, 2))

sim_lba <- param_infer_best %>%
  split(list(.$participant, .$list), drop = TRUE) %>%
  future_map_dfr(function (x) {
    bind_cols(filter(dlba_dat, participant == x$participant, list == x$list),
              density = dLBA(rt = filter(dlba_dat, participant == x$participant, list == x$list)$rt,
                             response = filter(dlba_dat, participant == x$participant, list == x$list)$response,
                             A = x$A,
                             b = x$b,
                             t0 = x$t0,
                             meanlog_v = c(x$meanlog_v1, x$meanlog_v2),
                             sdlog_v = c(1, x$sdlog_v2),
                             distribution = "lnorm",
                             silent = TRUE))
  }) %>%
  mutate(rt = ifelse(response == 1, rt, -rt),
         model = "LBA")
```

## Visualise fit

Plot LBA best fit over the distribution of the data from four
participants:

``` r
# Randomly sample 4 participants
set.seed(2021)
participant_sample <- sample(participants, 4)

d_sample <- filter(d, participant %in% participant_sample)
sim_lba_sample <- filter(sim_lba, participant %in% participant_sample)

d_sample$participant <- factor(d_sample$participant, levels = participant_sample, labels = c("Participant 1", "Participant 2", "Participant 3", "Participant 4"))
sim_lba_sample$participant <- factor(sim_lba_sample$participant, levels = participant_sample, labels = c("Participant 1", "Participant 2", "Participant 3", "Participant 4"))

d_sample$list <- factor(d_sample$list, levels = c(1, 2, 3), labels = c("Session 1", "Session 2", "Session 3"))
sim_lba_sample$list <- factor(sim_lba_sample$list, levels = c(1, 2, 3), labels = c("Session 1", "Session 2", "Session 3"))


trial_counts <- d_sample %>%
  group_by(participant, list, response) %>%
  summarise(n = n()) %>%
  group_by(participant, list) %>%
  summarise(accuracy = n[response == 1]/sum(n),
            n = sum(n)) %>%
  mutate(label = paste0("n = ", n, "\n", prettyNum(accuracy*100, digits = 3), "% correct"))


draw_key_custom <- function(data, params, size) {
  if(data$colour == "#000000" && data$size == .1) { # Data
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
  else if (data$colour == "#e66101" && data$size == 1.1) { # LBA
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


d_sample %>%
  mutate(rt = ifelse(response == 1, rt, -rt),
         model = "Data") %>%
  ggplot(aes(x = rt, colour = model)) +
  facet_grid(list ~ participant, drop = TRUE) +
  geom_vline(xintercept = 0, lty = 2, colour = "grey80") +
  geom_histogram(aes(y = ..density..), binwidth = 1, fill = "white", size = .1, key_glyph = draw_key_custom) +
  geom_line(data = sim_lba_sample, aes(y = density), size = rel(1.1), alpha = .8, key_glyph = draw_key_custom) +
  geom_text(data = trial_counts, aes(label = label), x = -20, y = .4, hjust = 0, colour = "black", size = rel(3), fontface = "italic") +
  scale_x_continuous(limits = c(-20, 20), breaks = c(-15, 0, 15)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_colour_manual(values = c("#000000", "#e66101")) +
  labs(x = "RT (s)",
       y = "Density",
       colour = NULL) +
  theme_paper +
  theme(axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        strip.background = element_blank(),
        strip.text = element_text(size = rel(1)),
        legend.background = element_blank(),
        legend.position = "top",
        legend.justification = "right",
        legend.direction = "vertical",
        legend.box.margin = unit(c(-20, -30, -20, 0), "pt"))
```

    ## Warning: Removed 3 rows containing non-finite values (stat_bin).

    ## Warning: Removed 24 rows containing missing values.

![](02_empirical_example_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "param_recov_real_dist.pdf"), width = 9, height = 5)
```

    ## Warning: Removed 3 rows containing non-finite values (stat_bin).
    
    ## Warning: Removed 24 rows containing missing values.

``` r
ggsave(file.path("..", "output", "param_recov_real_dist.png"), width = 9, height = 5, dpi = 600)
```

    ## Warning: Removed 3 rows containing non-finite values (stat_bin).
    
    ## Warning: Removed 24 rows containing missing values.

## Analyse ACT-R parameters

Plot distribution of parameter estimates per
session:

``` r
param_labs <- c("expression(mu[c])", "expression(mu[f])", "expression(sigma[f])", "expression(F)", "expression(t[er])")
names(param_labs) <- c("A_c", "A_f", "A_f_sd", "F", "t_er")


param_infer_plotdat <- param_infer_best %>%
  transmute(`F` = b - A/2,
            A_c = meanlog_v1,
            A_f = meanlog_v2,
            A_f_sd = sdlog_v2,
            t_er = t0,
            participant,
            list) %>%
  pivot_longer(`F`:t_er, "parameter", "value") %>%
  mutate(parameter = factor(parameter, 
                            levels = c("A_c", "A_f", "A_f_sd", "F", "t_er"),
                            labels  = c(expression(mu[c]), expression(mu[f]), expression(sigma[f]), expression(F), expression(t[er]))),
         list_jitter = jitter(list, .4)) %>%
  filter(participant %in% participants)


param_infer_summary <- param_infer_plotdat %>%
  group_by(parameter, list) %>%
  summarise(median = median(value))


ggplot(param_infer_plotdat, aes(x = list_jitter, y = value, group = participant, colour = parameter)) +
  facet_wrap(~ parameter, ncol = 5, scales = "free_y", labeller = labeller(parameter = label_parsed))+
  geom_line(alpha = .15) +
  geom_point(alpha = .25) +
  geom_line(data = param_infer_summary, aes(x = list, y = median, group = parameter), colour = "black", lty = 2) +
  geom_point(data = param_infer_summary, aes(x = list, y = median, group = parameter), colour = "black", size = rel(2.5)) +
  scale_x_continuous(breaks = c(1, 2, 3)) +
  scale_colour_viridis_d() +
  labs(x = "Session",
       y = "Parameter value") +
  guides(colour = FALSE) +
  theme_paper +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = rel(1)))
```

![](02_empirical_example_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "param_infer_real_values.pdf"), width = 9, height = 3)
ggsave(file.path("..", "output", "param_infer_real_values.png"), width = 9, height = 3, dpi = 600)
```

Were there significant changes in parameters from session to session?

``` r
param_infer_modeldat <- param_infer_plotdat %>%
  mutate(session = factor(list, levels = c(2, 1, 3))) %>%
  select(-list, -list_jitter) %>%
  pivot_wider(names_from = "parameter", values_from = "value") %>%
  mutate(mu_diff = `mu[c]` - `mu[f]`) %>%
  pivot_longer(F:mu_diff, "parameter", "value")

# Activation of correct answer
m_mu_c <- lmer(value ~ session + (1 | participant),
               data = filter(param_infer_modeldat, parameter == "mu[c]"))

summary(m_mu_c)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ session + (1 | participant)
    ##    Data: filter(param_infer_modeldat, parameter == "mu[c]")
    ## 
    ## REML criterion at convergence: 117.5
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.4132 -0.5592  0.0280  0.5542  3.6761 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.03819  0.1954  
    ##  Residual                0.09178  0.3030  
    ## Number of obs: 150, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##              Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)  -0.28285    0.05098 125.35146  -5.548 1.64e-07 ***
    ## session1     -0.16918    0.06059  98.00001  -2.792   0.0063 ** 
    ## session3     -0.03918    0.06059  98.00001  -0.647   0.5194    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.594       
    ## session3 -0.594  0.500

``` r
# Activation of incorrect answer
m_mu_f <- lmer(value ~ session + (1 | participant),
               data = filter(param_infer_modeldat, parameter == "mu[f]"))

summary(m_mu_f)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ session + (1 | participant)
    ##    Data: filter(param_infer_modeldat, parameter == "mu[f]")
    ## 
    ## REML criterion at convergence: 140.1
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.8940 -0.4542 -0.0123  0.6141  2.3568 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.02951  0.1718  
    ##  Residual                0.11606  0.3407  
    ## Number of obs: 150, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##              Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)  -1.48487    0.05396 135.83770 -27.520   <2e-16 ***
    ## session1      0.05028    0.06814  98.00000   0.738    0.462    
    ## session3     -0.01603    0.06814  98.00000  -0.235    0.814    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.631       
    ## session3 -0.631  0.500

``` r
# Difference in activation
m_mu_diff <- lmer(value ~ session + (1 | participant),
                  data = filter(param_infer_modeldat, parameter == "mu_diff"))

summary(m_mu_diff)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ session + (1 | participant)
    ##    Data: filter(param_infer_modeldat, parameter == "mu_diff")
    ## 
    ## REML criterion at convergence: 173.6
    ## 
    ## Scaled residuals: 
    ##      Min       1Q   Median       3Q      Max 
    ## -2.07489 -0.59028 -0.01924  0.49703  2.57003 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.08338  0.2888  
    ##  Residual                0.12120  0.3481  
    ## Number of obs: 150, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##              Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)   1.20202    0.06397 110.34198  18.792  < 2e-16 ***
    ## session1     -0.21946    0.06963  98.00000  -3.152  0.00215 ** 
    ## session3     -0.02315    0.06963  98.00000  -0.332  0.74024    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.544       
    ## session3 -0.544  0.500

``` r
# SD of activation of incorrect answer
m_sigma_f <- lmer(value ~ session + (1 | participant),
               data = filter(param_infer_modeldat, parameter == "sigma[f]"))

summary(m_sigma_f)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ session + (1 | participant)
    ##    Data: filter(param_infer_modeldat, parameter == "sigma[f]")
    ## 
    ## REML criterion at convergence: 38
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.9371 -0.4859  0.0020  0.5562  4.2621 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.01323  0.1150  
    ##  Residual                0.05898  0.2429  
    ## Number of obs: 150, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##               Estimate Std. Error         df t value Pr(>|t|)    
    ## (Intercept)   0.778795   0.038003 137.751278  20.493   <2e-16 ***
    ## session1     -0.007876   0.048572  98.000000  -0.162    0.872    
    ## session3     -0.013522   0.048572  98.000000  -0.278    0.781    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.639       
    ## session3 -0.639  0.500

``` r
# Latency factor
m_f <- lmer(value ~ session + (1 | participant),
               data = filter(param_infer_modeldat, parameter == "F"))

summary(m_f)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ session + (1 | participant)
    ##    Data: filter(param_infer_modeldat, parameter == "F")
    ## 
    ## REML criterion at convergence: 188.4
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -2.0697 -0.6743 -0.1022  0.6621  4.7049 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.02036  0.1427  
    ##  Residual                0.17639  0.4200  
    ## Number of obs: 150, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##              Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)   1.81728    0.06273 143.91886  28.971   <2e-16 ***
    ## session1     -0.12503    0.08400  97.99981  -1.488   0.1398    
    ## session3     -0.21335    0.08400  97.99981  -2.540   0.0127 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.670       
    ## session3 -0.670  0.500

``` r
# Non-retrieval time
m_t_er <- lmer(value ~ session + (1 | participant),
               data = filter(param_infer_modeldat, parameter == "t[er]"))

summary(m_t_er)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: value ~ session + (1 | participant)
    ##    Data: filter(param_infer_modeldat, parameter == "t[er]")
    ## 
    ## REML criterion at convergence: 27
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.5079 -0.4728  0.0010  0.7037  1.7486 
    ## 
    ## Random effects:
    ##  Groups      Name        Variance Std.Dev.
    ##  participant (Intercept) 0.01766  0.1329  
    ##  Residual                0.05130  0.2265  
    ## Number of obs: 150, groups:  participant, 50
    ## 
    ## Fixed effects:
    ##              Estimate Std. Error        df t value Pr(>|t|)    
    ## (Intercept)   1.05646    0.03714 129.95790  28.449   <2e-16 ***
    ## session1      0.03088    0.04530  98.00000   0.682   0.4970    
    ## session3     -0.10592    0.04530  98.00000  -2.338   0.0214 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) sessn1
    ## session1 -0.610       
    ## session3 -0.610  0.500

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
    ## [1] grid      stats4    stats     graphics  grDevices utils     datasets 
    ## [8] methods   base     
    ## 
    ## other attached packages:
    ##  [1] lmerTest_3.1-0  lme4_1.1-21     Matrix_1.2-18   rlang_0.4.10   
    ##  [5] cowplot_0.9.4   truncdist_1.0-2 evd_2.3-3       tidyr_1.0.0    
    ##  [9] furrr_0.1.0     future_1.13.0   purrr_0.3.2     rtdists_0.11-2 
    ## [13] ggplot2_3.3.2   dplyr_0.8.3    
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.1    xfun_0.21           listenv_0.7.0      
    ##  [4] splines_3.6.3       lattice_0.20-41     colorspace_1.4-1   
    ##  [7] vctrs_0.3.8         expm_0.999-4        viridisLite_0.3.0  
    ## [10] htmltools_0.3.6     yaml_2.2.0          survival_2.44-1.1  
    ## [13] nloptr_1.2.1        pillar_1.4.2        glue_1.3.1         
    ## [16] withr_2.3.0         lifecycle_0.1.0     stringr_1.4.0      
    ## [19] munsell_0.5.0       gtable_0.3.0        mvtnorm_1.1-1      
    ## [22] codetools_0.2-16    evaluate_0.14       labeling_0.3       
    ## [25] knitr_1.23          parallel_3.6.3      Rcpp_1.0.6         
    ## [28] scales_1.0.0        jsonlite_1.6        digest_0.6.19      
    ## [31] stringi_1.4.3       msm_1.6.8           numDeriv_2016.8-1.1
    ## [34] gsl_2.1-6           tools_3.6.3         magrittr_1.5       
    ## [37] tibble_2.1.3        crayon_1.3.4        pkgconfig_2.0.2    
    ## [40] MASS_7.3-51.4       minqa_1.2.4         assertthat_0.2.1   
    ## [43] rmarkdown_2.6       R6_2.4.0            globals_0.12.4     
    ## [46] boot_1.3-25         nlme_3.1-149        compiler_3.6.3
