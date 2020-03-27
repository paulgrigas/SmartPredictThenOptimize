library(tidyverse)
library(gridExtra)

results <- read_csv("portfolio.csv")

results = results %>% filter(grid_dim == 50 & p_features == 5 & polykernel_degree < 32) # grid_dim is actually the number of assets here

summary(results)

results <- results %>% mutate(SPOplus_norm_spo = -SPOplus_spoloss_test/zstar_avg_test,
                              LS_norm_spo = -LS_spoloss_test/zstar_avg_test,
                              RF_norm_spo = -RF_spoloss_test/zstar_avg_test,
                              Absolute_norm_spo = -Absolute_spoloss_test/zstar_avg_test)


# Processing for both plot

colnames(results)[colnames(results)=="SPOplus_norm_spo"] <- "SPO+"
colnames(results)[colnames(results)=="LS_norm_spo"] <- "Least Squares"
colnames(results)[colnames(results)=="RF_norm_spo"] <- "Random Forests"
colnames(results)[colnames(results)=="Absolute_norm_spo"] <- "Absolute Loss"
#colnames(results)[colnames(results)=="Huber_norm_spo"] <- "Huber Loss"


# correct processing 
results_relevant = results %>% 
  select(grid_dim, p_features, n_train, polykernel_degree, polykernel_noise_half_width,
         `SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`)

results_relevant_fixed = results_relevant %>%
  gather(`SPO+`, `Least Squares`, `Random Forests`, `Absolute Loss`,
         key = "method", value = "spo_normalized")

results_relevant_fixed$method = as.factor(results_relevant_fixed$method)
results_relevant_fixed$n_train = as.factor(results_relevant_fixed$n_train)
results_relevant_fixed$polykernel_noise_half_width = as.factor(results_relevant_fixed$polykernel_noise_half_width)
results_relevant_fixed$grid_dim = as.factor(results_relevant_fixed$grid_dim)
results_relevant_fixed$p_features = as.factor(results_relevant_fixed$p_features)


# Labelers
training_set_size_names <- c(
  '100' = "Training Set Size = 100",
  '1000' = "Training Set Size = 1000",
  '5000' = "Training Set Size = 5000"
)

half_width_names <- c(
  '1' = "Noise Factor = 1",
  '2' = "Noise Factor = 2" 
)

p_features_names <- c(
  '5' = "p = 5",
  '10' = "p = 10"
)

grid_dim_names <- c(
  '2' = "2 x 2 grid",
  '3' = "3 x 3 grid",
  '4' = "4 x 4 grid",
  '5' = "5 x 5 grid" 
)

training_set_size_labeller <- as_labeller(training_set_size_names)
half_width_labeller <- as_labeller(half_width_names)
p_features_labeller <- as_labeller(p_features_names)
grid_dim_labeller <- as_labeller(grid_dim_names)


####### BOX PLOT ####### 

plot <- results_relevant_fixed %>%
  ggplot(aes(x = as.factor(polykernel_degree), y = spo_normalized, fill = method)) +
  geom_boxplot() +
  scale_y_continuous(name = "Normalized SPO Loss", labels = scales::percent_format(accuracy = 1)) +
  scale_fill_discrete(name = "Method") +
  facet_wrap(vars(n_train, polykernel_noise_half_width), 
             labeller = labeller(n_train = training_set_size_labeller, polykernel_noise_half_width = half_width_labeller), 
             ncol = 2, scales = "free") + 
  theme_bw() +
  labs(x = "Polynomial Degree", title = "Normalized SPO Loss vs. Polynomial Degree") +
  theme(axis.title=element_text(size=36), axis.text=element_text(size=30), legend.text=element_text(size=36), 
        legend.title=element_text(size=36), strip.text = element_text(size = 24), 
        legend.position="top", plot.title = element_text(size = 42, hjust = 0.5))

plot

ggsave("portfolio_plot.pdf", width = 20, height = 12, units = "in")
