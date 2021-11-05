---
draft: false
title: "Enabling the energy transition with smart meter data"
date: 2021-11-03
authors:
- Jonas Soenen
- Lola Botman
- Konstantinos Theodorakos
- Dries Van Daele
- Aras Yurtman
- Jessa Bekker
- Koen Vanthournout
tags:
- energy forecasting
- smart grid
categories: ["Machine Learning & Data Science"]
social:
- icon: envelope
  icon_pack: fas
  link: 'about/#contact'  # For a direct email link, use "mailto:test@example.org".
featured: true
image:
  placement: 1
  caption: ""
  focal_point: "Center"
  preview_only: false
share: true
summary: "Increasing household electricity consumption (e.g. widespread adoption of electrical vehicles) is driving the low-voltage energy grid to its capacity limits. Simply replacing or scaling up such an infrastructure is neither trivial nor cost-effective. Instead, the Flanders AI Research Program, supports research on intelligent solutions for grid management. Here we present different AI-driven approaches currently under investigation by KU Leuven and EnergyVille."
draft: false
---

{{% relpub %}}
The Flanders AI Program, started by the Flemish government, focusses on AI research, implementation, ethics and education. With this program, the Flemish government wants to ensure that Flanders is ready for the AI evolution. The whole program represents a yearly budget of 32 million euros. The research portion of the program ([The Flanders AI Research program](https://www.flandersairesearch.be/en)) aims to strengthen the fundamental research that is already present in Flanders and encourages researchers to apply AI techniques to several use cases in healthcare, Industry 4.0 and government. The program consists of around 120 researchers and has already resulted in more than 100 publications in high quality journals and conferences. In this post, we explore the low voltage grid use case of the program and give a glimpse of the ongoing research.
{{% /relpub %}}

There is an immense energy transition underway as we move from fossil fuels to more renewable energy sources. More and more households become what we now call prosumers: they both consume and produce electricity (e.g. solar panels). In the past, electricity only flowed from the distribution grid to the household, but now, for prosumers, electricity flows from and to the household. In the meantime, the widespread adoption of electrical vehicles (EVs) is underway. The Belgian government is fostering this adoption: by 2026, they want emission-free company cars to be the norm and they are offering tax incentives for the installation of EV charging infrastructure.

This ongoing transition influences the way we use electricity. The widespread adoption of EVs, among other factors, is expected to cause a significant increase in electricity demand. Not only do we consume more electricity, we also consume differently, i.e. the electricity consumption patterns are changing. For example, on a sunny day, a neighborhood with lots of solar panels can generate significantly more electricity than what is consumed. Each household with solar panels injects its electricity surplus into the grid at the same time, causing a high simultaneous injection load on the distribution grid. Analogously, when people return from work, they plug in their EVs to charge. In some places with lots of EVs, this might again result in a high simultaneous load on the grid.

To meet the changing electricity consumption, the low voltage (LV) distribution grid will need to be reinforced in the near future and doing so cost efficiently is a major challenge. The electrical grid as we know it today was built over the course of the last century using a ‘fit and forget’ strategy: install over-dimensioned cables with sufficient capacity to cover all peaks in electricity demand. Historically, this was a very efficient method as the peak on a cable was typically only 20-30% of the sum of individual peaks of the houses connected to the cable: we do not have our individual highest consumption at the same time. However, as our consumption increases and becomes more synchronized, there is a higher risk that at some point the installed capacity doesn’t suffice and congestion starts to occur. That is, the voltage drops too low or rises too high, leading to appliance malfunctions, and ultimately too high currents blow the cable fuse and the lights go out. Upgrading the grid using the ‘fit and forget’ strategy to handle higher loads would be expensive. The scale of the LV distribution grid is massive; Flanders alone counts 84000km of cable and appr. 3,5 million connection points. To reinforce mere percentages already runs into the hundreds of millions Euro. The main challenge is to support the increasing load on the low voltage grid through a more optimal use of the capacity of the current grid assets and targeted grid reinforcement investments only where necessary.

This is one of the challenges that [EnergyVille](https://www.energyville.be/en) (a cooperation between [KU Leuven](https://www.kuleuven.be/english), [VITO](https://vito.be/en), [imec](https://www.imec-int.com/en) and  [UHasselt](https://www.uhasselt.be/en)) is tackling in collaboration with [Fluvius](https://www.fluvius.be/), the Flemish distribution system operator (DSO).


## Increasing the visibility in the low voltage grid

The first step towards more cost-effective grid management is to gain a detailed and accurate view on the current state of the LV grid. Obtaining more detailed information about the grid’s layout and state allows us to identify the problematic parts of the grid and to quantify the severity of these problems.

To achieve this, EnergyVille is developing ‘digital twin’ capabilities for the LV grid: a set of tools that yields a more detailed view on the grid by applying a combination of AI and Power System Engineering techniques on the available data. One of the central systems in this toolset is a simulation environment that, given the consumption of connected households and the layout of the LV grid, calculates all currents and voltages in the grid (i.e. how the electricity ‘flows’ through the grid). However, consumption measurements in the LV grid - although increasing - remain sparse and our knowledge about the layout of the grid is incomplete and often inaccurate. Therefore, a crucial aspect of the work is to develop tools that correct and enhance the available data.

{{< figure_caption fig="overview.png">}}

As one of the industrial use cases within the Flanders AI Research program, a collaboration was set up between DTAI-KU Leuven and STADIUS-KU Leuven research groups and EnergyVille to tackle some of the data quality issues and to take the first steps towards predicting problems in the grid ahead of time:
- How can we generate accurate consumption data for households with no or only few available consumption measurements to use in the grid simulation?
- Can we derive accurate maps of the low voltage grid based on available schematic drawings?
- Can we forecast congestion problems in the grid ahead of time?
- Can we reduce the time it takes to simulate the grid for a longer period of time, e.g., a year?


## Measuring the similarity between households and clustering them

To be able to simulate the grid, the system developed by EnergyVille needs quarter hour consumption data for every household. However, only for a small portion of households there is such consumption data available for a longer period of time. As such, the goal of this task is to accurately measure the similarity between different consumption profiles. Whenever a profile of a certain household doesn’t have enough data available, we can use consumption data of the most similar complete profiles instead.

The biggest challenge when measuring the similarity between energy consumption profiles is that the profiles themselves are very irregular and highly stochastic. For example, one day, a household might cook extensively using the stove, and on another, they may simply use the microwave. The exact time at which they cook might also differ from day to day. As such, comparing the corresponding timestamps of two profiles (e.g. the consumption measured on Jan 1 at 19:15 of profile 1 with the consumption on Jan 1 at 19:15 of profile 2) might consider two profiles highly dissimilar while both profiles actually behave similarly: they cook around the same time (e.g. plus/minus 30 minutes) and both use the stove or microwave to cook (but not always on the same day). Hence, we aim to develop a flexible similarity measure that takes the stochasticity of electricity consumption into account.

Based on such a flexible similarity metric, we can apply COBRAS, a clustering algorithm developed at DTAI-KU Leuven. COBRAS takes into account the domain experts knowledge by posing questions to the expert during the clustering process. Embedding this knowledge in the process ensures that the produced clustering is suited for the problem at hand. The resulting clustering allows EnergyVille to study the different types of energy consumers in Belgium and provides an easy way to sample consumption data to use for simulating the grid.

For more information see [the COBRAS webpage](https://dtai.cs.kuleuven.be/software/cobras/) or [a gentle introduction into using COBRAS for time series](https://dtai.cs.kuleuven.be/stories/post/wannes/time-series-clustering/).

## Deriving accurate maps of the low voltage grid from schematics

In addition to consumption data, a simulation of the low voltage grid also requires an accurate map of the grid layout. It is crucial to know which houses are connected to which cables; when many houses attached to the same part of the grid simultaneously experience a peak in energy consumption, that part of the grid is at risk for congestion. Furthermore, to accurately calculate the voltage drop or rise , the length of the cables is required to be known.

Unfortunately, only inaccurate schematic drawings of the LV grid are available. These schematics were historically intended for human interpretation. Because of this, they display inaccuracies that can generally be resolved using common sense. When multiple cables run alongside the same side of the road, they are depicted as visually distinct lines, which greatly exaggerates their true distance from one another. Near transformers, where many cables gather, the depiction becomes particularly chaotic. Furthermore, important information such as the knowledge of which houses are connected to which cable is left implicit, to be inferred by the reader. An unfortunate side-effect of these visual simplifications and abstractions is that naive, automated usage of this schematic data is likely to result in some invalid beliefs about the low voltage grid layout.

{{< figure_caption fig="gis.png" caption="Some example synthetic data as it might appear in a schematic, where the house-to-cable connections have been manually added.">}}

We are developing novel methods to enhance the available schematic drawings and to derive a more accurate map of the low voltage grid with correct house-to-cable connections. To capture the real-world context with its numerous roads and houses, we enrich the schematics with public geographical information from [Geopunt Vlaanderen](https://www.geopunt.be/). While all of this is raw geographical (GIS) data consisting of simple points and lines in a 2D-space, the challenge is to facilitate learning models that capture the visual common sense that underlies the schematics. This is achieved by introducing novel geographical primitives that capture the visual relations between objects, rather than just their coordinates. Examples of such primitives are: “points A and B are on the same side of line C” or “there is no obstruction between point A and B”. Machine learning methods can build upon these concepts to learn more complex and qualitative models, without having to deal with the raw data.


## Forecasting households’ consumption 24 hours ahead

One way to forecast congestion problems in the LV grid is to forecast each connected households’ consumption and feed this forecasted consumption into the simulation environment. If the forecast indicates that there might be congestion tomorrow, action can be taken in an effort to prevent it (for example, sending appropriate control signals to devices such as smart heat pumps, smart charging of EVs, home batteries, etc. in order to change their load).  

However, as mentioned in the clustering task, the consumption profiles are very irregular and stochastic and therefore difficult to forecast. Energy consumption profiles are usually characterized by sudden short high electrical consumption periods, appearing as peaks in the consumption profiles. To forecast congestion accurately it is essential to know when a demand peak will take place and how large the peak will be. However, conventional regression algorithms such as Auto Regressive Integrated Moving Average (ARIMA) and Long Short Term Memory (LSTM) networks struggle to predict these peaks because the input data is highly imbalanced (i.e. lot of low values and only a few high values).

{{< figure_caption fig="ts.png">}}

To sidestep the issues around the imbalance of the input data, we reframe the forecasting (regression) problem as a classification problem. In a regression problem, the algorithm will predict the next value, while in a classification problem, the algorithm will predict whether or not the next value is a peak value. In the reframed problem there will still be significantly more lower values than peak values but in a classification setting there are several approaches to resolve this imbalance. In the next phase, to also predict the size of the peak, we could opt for a multi-class classification problem where a peak value is also classified as low peak, high peak, very high peak, etc… This change in perspective is currently being explored and, if successful, would allow us to increase the congestion prediction performances.

{{< figure_caption fig="regress_class.png">}}


## Robust days-of-year clustering of smart-meter data

As discussed earlier, grid simulations can support distribution system operators in making informed long term strategic decisions. However, performing all the required calculations that simulate the intricate parts of the low voltage grid takes time. Simulating the grid for short periods of time is feasible (e.g. a single day of simulation requires around 14 hours of computation), but simulating for longer periods of time (e.g. a year) becomes problematic in terms of the calculation time. While single-day simulations are useful in assessing specific “worst-case estimates”, full-year simulations are necessary to get a good overall idea about the frequency and severity of congestion problems.

Luckily, there might be a way around this problem. A year contains days that are similar to each other, both meteorologically and in energy production/consumption, e.g.,days in the same season, weekends/holidays. As such, instead of simulating every day of the year, we can perform the simulation for a limited number of representative days. This strategy reduces the calculation time of a full-year simulation considerably.  However, in order for this to work, we need a way to select the representative days such that the quality of the reduced simulation is as close as possible to the difficult-to-calculate full-year simulation.

{{< figure_caption fig="im5.png">}}

To select the representative days, we cluster the days of the year based on household electricity consumption and meteorological (e.g. temperature, rainfall, …) time series data; the medoids (i.e. centers) of each cluster are selected as representative days. After preprocessing the timeseries to remove erroneous values and impute missing values, matrix decomposition reduces the dimensionality of the problem and robust k-medoids performs the clustering itself. The (hyper) parameters of the matrix decomposition  and the k-medoids algorithm are optimized through bayesian optimization. A schematic overview of our approach is shown in the figure above.

## The low voltage grid: from barrier to enabler of the energy transition
Without realizing it, we use the LV grid every day; our houses, offices, and schools are connected to it. Each time we leave or enter a building, we pass over or under its myriad of cables. It is a critical infrastructure without which our civilization would fail to function. The various disaster books and movies on nationwide blackouts attest to this. Yet we rarely think of the grid. We have almost forgotten that it is there.

This seemingly boring collection of copper and aluminum wires finds itself at the forefront to stop climate change. Renewable electricity is the most important alternative to fossil fuels. This electrical ‘fuel’ that powers our cars and heating systems must now pass over the LV grid. Heating a house with a heat pump and charging the owner’s electrical car roughly triples the household’s electrical consumption and on a sunny day that same house injects its excess local production into the grid. Handling this increased consumption and injection is a formidable challenge that shakes the system. Not in the future, but already today; simply check a random newspaper to see the many articles discussing electric cars, support systems for solar panels, cost of charging infrastructure, etc. The grid is no longer invisible.

And so, that is our goal: to help make the LV grid invisible once more. To ensure that we have the technology to minimize the cost of upgrading the grid to support renewable electricity production and the electrification of fossil fuel consumers. To ensure that we have all the data, forecasts and control signals such that our shiny new green devices can respond automatically to the availability of renewable energy while respecting the transport capacity of the local grid. To ensure that the LV grid is no bottleneck for the energy transition but an enabler of it. So we can continue to live our lives in comfort, totally oblivious of what is behind those power sockets in the wall.
