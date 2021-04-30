## Shiny app hosted at https://mavdvelde.shinyapps.io/actr-lba/
## Author: Maarten van der Velde

library(shiny)
library(dplyr)
library(ggplot2)
library(rtdists)
library(grid)
library(rlang)

theme_shiny <- theme_classic(base_size = 16) + 
  theme(axis.text = element_text(colour = "black"))


ui <- fluidPage(
  
  # Application title
  titlePanel("ACT-R/LBA simulation"),
  
  # Description
  HTML("This page provides an interactive version of Figure 1 in <i> \
    van der Velde, M., Sense, F., Borst, J., & van Rijn, H. (2021). \
    Capturing Dynamic Performance in a Cognitive Model: Estimating ACT-R Memory Parameters with the Linear Ballistic Accumulator. </i>"),
  
  hr(),
  
  # Sidebar with slider inputs 
  sidebarLayout(
    sidebarPanel(
      h4("Set model parameters"),
      sliderInput("latency_factor",
                  HTML("Latency factor <i>F</i>:"),
                  min = 0.5,
                  max = 5,
                  value = 2,
                  step = .1),
      sliderInput("t_er",
                  HTML("Non-retrieval time <i>t<sub>er</sub></i>:"),
                  min = 0,
                  max = 5,
                  value = .5,
                  step = .1),
      sliderInput("a_c_mu",
                  HTML("Mean activation of correct answer <i>&mu;<sub>c</sub></i>:"),
                  min = -5,
                  max = 5,
                  value = -.5,
                  step = .1),
      sliderInput("a_c_sd",
                  HTML("SD of activation of correct answer <i>&sigma;<sub>c</sub></i>:"),
                  min = 0,
                  max = 5,
                  value = 1,
                  step = .1),
      sliderInput("a_f_mu",
                  HTML("Mean activation of incorrect answer <i>&mu;<sub>i</sub></i>:"),
                  min = -5,
                  max = 5,
                  value = -1.5,
                  step = .1),
      sliderInput("a_f_sd",
                  HTML("SD of activation of incorrect answer <i>&sigma;<sub>i</sub></i>:"),
                  min = 0,
                  max = 5,
                  value = 1.5,
                  step = .1),
      sliderInput("n",
                  "Trials:",
                  min = 1000,
                  max = 1e5,
                  value = 1e4,
                  step = 1000),
      actionButton("reset", "Reset sliders", width = "100%")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      h4("ACT-R model"),
      plotOutput("actr_plot"),
      h4("Comparison of RT distributions"),
      plotOutput("model_comp_plot")
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  
  observeEvent(input$reset, {
    updateSliderInput(session, "latency_factor", value = 2)
    updateSliderInput(session, "t_er", value = .5)
    updateSliderInput(session, "a_c_mu", value = -.5)
    updateSliderInput(session, "a_c_sd", value = 1)
    updateSliderInput(session, "a_f_mu", value = -1.5)
    updateSliderInput(session, "a_f_sd", value = 1.5)
    updateSliderInput(session, "n", value = 1e4)
    
  })
  
  
  output$actr_plot <- renderPlot({
    
    n_trials <- input$n
    
    sim_actr <- tibble(
      f = rep(input$latency_factor, n_trials),
      a_c = rnorm(n_trials, mean = input$a_c_mu, sd = input$a_c_sd),
      a_f = rnorm(n_trials, mean = input$a_f_mu, sd = input$a_f_sd),
      t = rep(input$t_er, n_trials)
    ) %>%
      rowwise() %>%
      mutate(rt = f * exp(-max(a_c, a_f)) + t,
             response = ifelse(a_c > a_f, 1, 2)) %>%
      ungroup()
    
    
    sim_actr_sample <- sample_n(sim_actr, 150)
    
    ggplot() +
      
      # t_er rectangle
      geom_rect(aes(xmax = input$t_er, xmin = 0, ymin = 0, ymax = 1), fill = "grey80") +
      geom_segment(aes(xend = input$t_er, x = 0, y = .5, yend = .5), lty = 1,
                   arrow = arrow(ends = "both", type = "closed", length = unit(2, "mm"))) +
      geom_text(aes(x = input$t_er/2, y = .7, label = paste0(expression(t[er]))), parse = TRUE, size = rel(5)) +
      
      # Top density plot
      geom_density(data = sim_actr,
                   aes(x = rt, y = after_stat(count)/(nrow(sim_actr)/2) + input$latency_factor + .5, colour = as.factor(response)),
                   n = 2^10,
                   adjust = .05) +
      
      # Dotted y-axis
      geom_vline(xintercept = 0, lty = 3) +
      
      # F arrow
      geom_segment(aes(x = 0, xend = 0, y = .5, yend = input$latency_factor + .5),
                   arrow = arrow(ends = "both", type = "closed", length = unit(2, "mm"))) +
      geom_text(aes(y = input$latency_factor/2 + .5, x = .2, label = "F"), size = rel(5)) +
      
      # Example trajectories
      geom_point(data = sim_actr_sample, aes(x = rt, y = f + .5, colour = as.factor(response)), alpha = .25) +
      geom_segment(data = sim_actr_sample, aes(x = t, xend = rt, y = .5, yend = f + .5, colour = as.factor(response)), alpha = .1, lwd = rel(.5)) +
      
      # Mean drift rate lines
      geom_segment(aes(x = input$t_er, xend = input$latency_factor/exp(input$a_c_mu) + input$t_er, yend = input$latency_factor + .5), y = .5, colour = "#0571b0", lty = 1) +
      geom_path(aes(x = (input$latency_factor/exp(input$a_c_mu)) / 2 + input$t_er + c(0, .4, .4, 0),
                    y = input$latency_factor/2 + .5 + c(0, 0, exp(input$a_c_mu)*.4, 0))) +
      geom_label(aes(x = (input$latency_factor/exp(input$a_c_mu)) / 2 + input$t_er + .725, y = input$latency_factor/2 + .5 + exp(input$a_c_mu)*.2,
                     label = paste0(expression(e^mu[c]))),
                 parse = TRUE, label.size = NA, size = rel(5), label.padding = unit(.1, "lines")) +
      
      geom_segment(aes(x = input$t_er, xend = input$latency_factor/exp(input$a_f_mu) + input$t_er, yend = input$latency_factor + .5), y = .5, colour = "#ca0020", lty = 1) +
      geom_path(aes(x = (input$latency_factor/exp(input$a_f_mu)) / 2 + input$t_er + c(0, .4, .4, 0),
                    y = input$latency_factor/2 + .5 + c(0, 0, exp(input$a_f_mu)*.4, 0))) +
      geom_label(aes(x = (input$latency_factor/exp(input$a_f_mu)) / 2 + input$t_er + .725, y = input$latency_factor/2 + .5 + exp(input$a_f_mu)*.2,
                     label = paste0(expression(e^mu[f]))),
                 parse = TRUE, label.size = NA, size = rel(5), label.padding = unit(.1, "lines")) +
      
      
      # Boundary line
      geom_hline(yintercept = input$latency_factor + .5) +
      
      scale_x_continuous(expand = c(.0075,0)) +
      coord_cartesian(xlim = c(0, 12), clip = "off") +
      scale_y_continuous(expand = c(0,0), limits = c(0, NA),
                         breaks = c(0, .5, 1, input$latency_factor + .5),
                         labels = c(0, "A/2", "A", "d")) +
      scale_colour_manual(values = c("#0571b0", "#ca0020")) +
      labs(x = "Time",
           y = NULL) +
      guides(colour = FALSE) +
      theme_shiny +
      theme(axis.ticks.y = element_blank(),
            axis.line.y = element_blank(),
            axis.text.y = element_text(colour = "grey50"))
    
  })
  
  
  output$model_comp_plot <- renderPlot({
    
    n_trials <- input$n
    
    sim_actr <- tibble(
      f = rep(input$latency_factor, n_trials),
      a_c = rnorm(n_trials, mean = input$a_c_mu, sd = input$a_c_sd),
      a_f = rnorm(n_trials, mean = input$a_f_mu, sd = input$a_f_sd),
      t = rep(input$t_er, n_trials)
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
    b <- input$latency_factor + .5*A
    
    dlba_dat <- expand.grid(rt = seq(0, 20, by = .01),
                            response = c(1, 2))
    
    sim_lba <- bind_cols(dlba_dat,
                         density = dLBA(rt = dlba_dat$rt,
                                        response = dlba_dat$response,
                                        A = A,
                                        b = b,
                                        t0 = input$t_er,
                                        meanlog_v = c(input$a_c_mu, input$a_f_mu),
                                        sdlog_v = c(input$a_c_sd, input$a_f_sd),
                                        distribution = "lnorm",
                                        silent = TRUE)) %>%
      mutate(rt = ifelse(response == 1, rt, -rt),
             model = "LBA")
    
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
    
    ggplot(sim_actr, aes(x = rt, colour = model)) +
      geom_vline(xintercept = 0, lty = 2, colour = "grey80") +
      geom_histogram(aes(y = ..density..), binwidth = .25, fill = "white", key_glyph = draw_key_custom) +
      geom_line(data = sim_lba, aes(y = density), size = rel(2), alpha = .8, key_glyph = draw_key_custom) +
      scale_x_continuous(limits = c(-20, 20)) +
      scale_y_continuous(expand = c(0, 0)) +
      scale_colour_manual(values = c("#000000", "#e66101")) +
      labs(x = "RT (s)",
           y = "Density",
           colour = NULL) +
      theme_shiny +
      theme(axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.position = c(.9, .9))
  })
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)
