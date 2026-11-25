# ADS 506 - Assignment 5.1: Australian Wine Sales Forecast Explorer
# Christopher Guillen
#
# AI disclosure: I used Claude (Anthropic) to help draft the first version of
# this app. I tested it, fixed issues, and adjusted it from there.

library(shiny)
library(fpp3)
library(urca)

# ---- Data prep (only runs once when the app starts up) ----------------------

raw <- readr::read_csv("AustralianWines.csv", show_col_types = FALSE)

wine <- raw |>
  mutate(
    Month = yearmonth(my(Month)),         # dates come in like "Jan-80"
    Rose  = suppressWarnings(as.numeric(Rose))  # Rose has "*" for Jul/Aug 1994, so those turn into NA
  ) |>
  pivot_longer(-Month, names_to = "Varietal", values_to = "Sales") |>
  as_tsibble(index = Month, key = Varietal)

# ETS can't handle gaps, so fill in the two missing Rose months using an
# ARIMA fit (same approach as fpp3 ch. 13.9), then put Rose back with the rest
rose_filled <- wine |>
  filter(Varietal == "Rose") |>
  model(ARIMA(Sales)) |>
  interpolate(wine |> filter(Varietal == "Rose"))

wine <- wine |>
  filter(Varietal != "Rose") |>
  bind_rows(rose_filled)

varietals  <- sort(unique(wine$Varietal))
first_date <- as.Date(min(wine$Month))
last_date  <- as.Date(max(wine$Month))

# the three models - stored in a list so the checkboxes can pick which ones to run
model_specs <- list(
  TSLM  = TSLM(Sales ~ trend() + season()),
  ETS   = ETS(Sales),
  ARIMA = ARIMA(Sales)
)

# translates things like ETS(M,A,M) or ARIMA(1,0,1)(0,1,1)[12] into plain
# English so the spec table makes sense to someone who isn't a stats person
describe_spec <- function(spec) {
  if (is.na(spec) || spec == "NULL model") return("Model failed to fit")
  if (startsWith(spec, "TSLM")) return("Linear regression on a trend + monthly dummies")
  if (startsWith(spec, "ETS")) {
    word <- c(N = "none", A = "additive", Ad = "damped additive", M = "multiplicative")
    p <- regmatches(spec, regexec("ETS\\(([^,]+),([^,]+),([^)]+)\\)", spec))[[1]]
    return(sprintf("Error: %s | Trend: %s | Seasonality: %s",
                   word[p[2]], word[p[3]], word[p[4]]))
  }
  if (startsWith(spec, "ARIMA")) {
    n <- as.integer(regmatches(spec, gregexpr("[0-9]+", spec))[[1]])
    out <- sprintf("Non-seasonal: AR %d, differenced %d, MA %d", n[1], n[2], n[3])
    if (length(n) >= 7) {
      out <- paste0(out, sprintf(" | Seasonal (every %d months): AR %d, differenced %d, MA %d",
                                 n[7], n[4], n[5], n[6]))
    }
    if (grepl("drift", spec)) out <- paste0(out, " | with drift")
    return(out)
  }
  spec
}

# ---- UI: sidebar controls on the left, tabs on the right --------------------

ui <- fluidPage(
  titlePanel("Australian Wine Sales: Forecast Model Explorer"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      selectizeInput("varietal", "Varietal(s)",
                     choices = varietals, selected = c("sparkling", "Red"),
                     multiple = TRUE),
      helpText("Pick one or more wine types. Each one gets its own models."),
      
      sliderInput("range", "Date range used",
                  min = first_date, max = last_date,
                  value = c(first_date, last_date), timeFormat = "%b %Y"),
      
      sliderInput("train_end", "Training ends (train/validation split)",
                  min = first_date + 365 * 3, max = last_date - 180,
                  value = as.Date("1992-12-01"), timeFormat = "%b %Y"),
      helpText("Models are fit on data up to this month and scored on the months after it."),
      
      checkboxGroupInput("models", "Models",
                         choices = names(model_specs), selected = names(model_specs),
                         inline = TRUE),
      
      sliderInput("h", "Forecast horizon (months)", min = 6, max = 60, value = 24, step = 6),
      
      radioButtons("level", "Prediction interval", choices = c("80%" = 80, "95%" = 95),
                   selected = 95, inline = TRUE)
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        tabPanel("Overview",
                 br(),
                 p("Monthly sales for the selected varietals. Blue = training window, orange = validation window."),
                 plotOutput("overview_plot", height = "500px")),
        
        tabPanel("Seasonality",
                 br(),
                 p("STL decomposition: splits each series into trend, a repeating yearly pattern, and leftover noise."),
                 plotOutput("stl_plot", height = "auto")),
        
        tabPanel("Models & Accuracy",
                 br(),
                 h4("Best model per varietal (lowest validation RMSE)"),
                 tableOutput("best_table"),
                 h4("Model specifications (auto-selected)"),
                 tableOutput("spec_table"),
                 h4(textOutput("train_label")),
                 tableOutput("train_acc"),
                 h4(textOutput("valid_label")),
                 tableOutput("valid_acc"),
                 helpText("RMSE/MAE are in cases of wine; MAPE is percent error. Lower is better.
                          Validation scores matter most since the models never saw that data.")),
        
        tabPanel("Forecasts",
                 br(),
                 p("Each column is a model, each row a varietal. Shaded band = prediction interval;
                   black line = actual sales."),
                 plotOutput("fc_plot", height = "auto")),
        
        tabPanel("About",
                 br(),
                 h4("Data"),
                 p("AustralianWines.csv: monthly sales of six Australian wine varietals, Jan 1980 to Dec 1994."),
                 p("Rose had two missing months (Jul and Aug 1994, recorded as '*'). They were filled with
                   ARIMA-based interpolation before any modeling."),
                 h4("Modeling"),
                 tags$ul(
                   tags$li("TSLM: linear trend + monthly seasonal dummies."),
                   tags$li("ETS: auto-selected exponential smoothing (error, trend, seasonal form)."),
                   tags$li("ARIMA: auto-selected seasonal ARIMA (p,d,q)(P,D,Q)[12].")
                 ),
                 p("Models are fit on the training window only and evaluated on the validation window
                   (all months after the split, within the selected date range)."),
                 h4("Reproducing figures"),
                 p("Every figure depends only on the sidebar settings. To reproduce one, set the same
                   varietals, date range, training end, models, horizon, and interval level.")
        )
      )
    )
  )
)

# ---- Server -----------------------------------------------------------------

server <- function(input, output, session) {
  
  # if the date range changes, move the split slider so it still makes sense:
  # need at least 3 years to train on and 6 months left over to validate
  observeEvent(input$range, {
    lo <- as.Date(yearmonth(input$range[1]) + 36)   # yearmonth math is in months, so +36 = 3 yrs
    hi <- as.Date(yearmonth(input$range[2]) - 6)    # leave at least 6 months for validation
    val <- as.Date(yearmonth(min(max(input$train_end, lo), hi)))
    updateSliderInput(session, "train_end", min = lo, max = hi, value = val,
                      timeFormat = "%b %Y")
  })
  
  # grab all the inputs together and wait half a second after the user stops
  # clicking before refitting, otherwise dragging a slider refits over and over
  settings <- reactive({
    list(varietal = input$varietal, range = input$range, train_end = input$train_end,
         models = input$models, h = input$h, level = as.numeric(input$level))
  }) |> debounce(500)
  
  # just the varietals and months the user picked
  filtered <- reactive({
    s <- settings()
    validate(need(length(s$varietal) > 0, "Pick at least one varietal."))
    wine |>
      filter(Varietal %in% s$varietal,
             Month >= yearmonth(s$range[1]),
             Month <= yearmonth(s$range[2]))
  })
  
  # cut the data at the training end date: before = train, after = validation
  split <- reactive({
    te  <- yearmonth(settings()$train_end)
    dat <- filtered()
    train <- dat |> filter(Month <= te)
    valid <- dat |> filter(Month > te)
    validate(
      need(n_distinct(train$Month) >= 36, "Training window needs at least 3 years. Widen the date range or move the split later."),
      need(n_distinct(valid$Month) >= 1, "No validation months left. Move the split earlier.")
    )
    list(train = train, valid = valid, train_end = te)
  })
  
  # fit only the checked models, and only on the training data
  fit <- reactive({
    s <- settings()
    validate(need(length(s$models) > 0, "Pick at least one model."))
    withProgress(message = "Fitting models...", {
      split()$train |> model(!!!model_specs[s$models])
    })
  })
  
  # two sets of forecasts: one over the validation months for scoring,
  # and one h months out for the forecast plot
  fc_valid  <- reactive(fit() |> forecast(new_data = split()$valid))
  fc_future <- reactive(fit() |> forecast(h = settings()$h))
  
  acc_train <- reactive({
    accuracy(fit()) |>
      select(Varietal, Model = .model, RMSE, MAE, MAPE) |>
      arrange(Varietal, RMSE)
  })
  
  acc_valid <- reactive({
    accuracy(fc_valid(), filtered()) |>
      select(Varietal, Model = .model, RMSE, MAE, MAPE) |>
      arrange(Varietal, RMSE)
  })
  
  # ---- Outputs (each one fills a placeholder from the UI) ----
  
  output$overview_plot <- renderPlot({
    te <- split()$train_end
    filtered() |>
      mutate(Window = if_else(Month <= te, "Training", "Validation")) |>
      ggplot(aes(x = Month, y = Sales, colour = Window, group = Varietal)) +
      geom_line() +
      facet_wrap(~Varietal, scales = "free_y", ncol = 2) +
      scale_colour_manual(values = c(Training = "steelblue", Validation = "darkorange")) +
      labs(y = "Sales (cases)", x = NULL) +
      theme_minimal(base_size = 14) +
      theme(legend.position = "bottom")
  })
  
  # one row per varietal, one panel per STL piece, each with its own y-axis
  # (Rose sells ~100 cases and sparkling ~6000, so shared axes would hide Rose)
  output$stl_plot <- renderPlot({
    filtered() |>
      model(STL(Sales ~ season(window = "periodic"))) |>
      components() |>
      as_tibble() |>
      select(Varietal, Month, Sales, trend, season_year, remainder) |>
      pivot_longer(c(Sales, trend, season_year, remainder), names_to = "Component") |>
      mutate(Component = factor(Component, levels = c("Sales", "trend", "season_year", "remainder"))) |>
      ggplot(aes(x = Month, y = value)) +
      geom_line(colour = "steelblue") +
      facet_wrap(~ Varietal + Component, scales = "free_y", ncol = 4) +
      labs(x = NULL, y = NULL) +
      theme_minimal(base_size = 12)
  }, height = function() 220 * max(1, length(settings()$varietal)))
  
  output$spec_table <- renderTable({
    f <- fit()
    mods <- intersect(names(model_specs), names(f))
    purrr::map_dfr(mods, function(m) {
      specs <- purrr::map_chr(f[[m]], fabletools::model_sum)
      tibble(Varietal = f$Varietal, Model = m, Specification = specs,
             `What it means` = purrr::map_chr(specs, describe_spec))
    }) |>
      arrange(Varietal)
  })
  
  output$train_label <- renderText({
    sp <- split()
    paste0("Training accuracy (", format(min(sp$train$Month)), " to ", format(sp$train_end), ")")
  })
  
  output$valid_label <- renderText({
    sp <- split()
    paste0("Validation accuracy (", format(min(sp$valid$Month)), " to ", format(max(sp$valid$Month)), ")")
  })
  
  output$train_acc <- renderTable(acc_train(), digits = 1)
  output$valid_acc <- renderTable(acc_valid(), digits = 1)
  
  output$best_table <- renderTable({
    acc_valid() |>
      group_by(Varietal) |>
      slice_min(RMSE, n = 1) |>
      ungroup() |>
      select(Varietal, `Best model` = Model, `Validation RMSE` = RMSE)
  }, digits = 1)
  
  output$fc_plot <- renderPlot({
    sp <- split()
    # only show ~3 years of actual data so the forecasts are still easy to see
    recent <- filtered() |> filter(Month > sp$train_end - 36)
    fc_future() |>
      autoplot(recent, level = settings()$level) +
      facet_grid(Varietal ~ .model, scales = "free_y") +
      labs(y = "Sales (cases)", x = NULL) +
      theme_minimal(base_size = 13) +
      theme(legend.position = "bottom")
  }, height = function() 280 * max(1, length(settings()$varietal)))
}

shinyApp(ui, server)
