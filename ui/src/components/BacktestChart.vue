<template>
  <div class="bg-white rounded-2xl shadow-lg p-4">
    <canvas ref="chartCanvas"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import Chart from 'chart.js/auto'

const props = defineProps({
  results: Object,
  benchmark: Object,
  multiStrategyResults: Object
})
const chartCanvas = ref(null)
let chartInstance = null

function renderChart() {
  if (!props.results || !chartCanvas.value) return

  if (chartInstance) chartInstance.destroy()

  const datasets = []
  
  // Define colors for multiple strategies
  const strategyColors = [
    'rgb(59, 130, 246)',   // Blue
    'rgb(239, 68, 68)',    // Red
    'rgb(34, 197, 94)',    // Green
    'rgb(168, 85, 247)',   // Purple
    'rgb(245, 158, 11)',   // Amber
    'rgb(236, 72, 153)',   // Pink
    'rgb(14, 165, 233)',   // Sky
    'rgb(139, 69, 19)',    // Brown
  ]

  // Check if we have multiple strategies to display
  if (props.multiStrategyResults && props.multiStrategyResults.strategies.length > 1) {
    // Add all strategy datasets
    props.multiStrategyResults.strategies.forEach((strategy, index) => {
      const color = strategyColors[index % strategyColors.length]
      
      datasets.push({
        label: strategy.strategy_name || `Strategy ${index + 1}`,
        data: strategy.portfolio_values,
        borderColor: color,
        backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
        fill: false,
        tension: 0.1
      })
    })
  } else {
    // Single strategy - original behavior
    datasets.push({
      label: props.results.strategy_name || 'Strategy Total Value',
      data: props.results.portfolio_values,
      borderColor: 'rgb(59, 130, 246)', // Blue
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      fill: false,
      tension: 0.1
    })

    // Add cash dataset if available for single strategy
    if (props.results.cash_values) {
      datasets.push({
        label: 'Strategy Cash',
        data: props.results.cash_values,
        borderColor: 'rgb(34, 197, 94)', // Green
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        fill: false,
        tension: 0.1,
        borderDash: [3, 3] // Dotted line for cash
      })
    }
  }

  // Add benchmark dataset if available
  if (props.benchmark && props.benchmark.portfolio_values) {
    datasets.push({
      label: 'Benchmark (Buy & Hold)',
      data: props.benchmark.portfolio_values,
      borderColor: 'rgb(107, 114, 128)', // Gray
      backgroundColor: 'rgba(107, 114, 128, 0.1)',
      fill: false,
      tension: 0.1,
      borderDash: [5, 5] // Dashed line for benchmark
    })
  }

  // Use dates from multi-strategy results if available, otherwise from single results
  const dates = props.multiStrategyResults ? 
    props.multiStrategyResults.strategies[0]?.dates || props.results.dates :
    props.results.dates

  chartInstance = new Chart(chartCanvas.value, {
    type: 'line',
    data: {
      labels: dates,
      datasets: datasets
    },
    options: {
      responsive: true,
      plugins: {
        legend: { 
          display: true,
          position: 'top'
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': $' + context.parsed.y.toLocaleString()
            }
          }
        }
      },
      scales: {
        x: { 
          title: { display: true, text: 'Date' },
          grid: { display: false }
        },
        y: { 
          title: { display: true, text: 'Portfolio Value ($)' },
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString()
            }
          }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    }
  })
}

onMounted(renderChart)
watch(() => [props.results, props.benchmark, props.multiStrategyResults], renderChart)
</script>
