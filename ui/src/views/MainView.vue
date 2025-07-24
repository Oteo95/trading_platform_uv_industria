<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center py-8 px-4">
    <h1 class="text-3xl font-semibold mb-4">Portfolio Backtesting (Multi-Strategy)</h1>
    <p class="mb-6 text-gray-600 max-w-xl text-center">
      Select strategies to compare and click "Run Backtest"!
    </p>

    <!-- Strategy Selection -->
    <div v-if="availableStrategies.length > 0" class="mb-8 p-6 bg-white rounded-lg shadow-md w-full max-w-2xl">
      <h2 class="text-xl font-semibold mb-4">Select Strategies to Test</h2>
      <div class="mb-4">
        <label for="strategy-select" class="block text-sm font-medium text-gray-700 mb-2">
          Choose strategies to compare:
        </label>
        <select 
          id="strategy-select"
          v-model="selectedStrategies" 
          multiple 
          class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
          style="min-height: 120px;"
        >
          <option 
            v-for="strategy in availableStrategies" 
            :key="strategy.endpoint" 
            :value="strategy.endpoint"
            class="py-2"
          >
            {{ strategy.name }} - {{ strategy.description }}
          </option>
        </select>
        <div class="mt-2 text-sm text-gray-500">
          Hold Ctrl/Cmd to select multiple strategies. Selected: {{ selectedStrategies.length }} strategy(ies)
        </div>
      </div>
      
      <!-- Selected Strategies Display -->
      <div v-if="selectedStrategies.length > 0" class="mt-4">
        <h3 class="text-sm font-medium text-gray-700 mb-2">Selected Strategies:</h3>
        <div class="flex flex-wrap gap-2">
          <span 
            v-for="endpoint in selectedStrategies" 
            :key="endpoint"
            class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
          >
            {{ getStrategyName(endpoint) }}
            <button 
              @click="removeStrategy(endpoint)"
              class="ml-2 text-blue-600 hover:text-blue-800"
            >
              ×
            </button>
          </span>
        </div>
      </div>
    </div>

    <button
      :disabled="loading || !pricesData || selectedStrategies.length === 0"
      @click="runBacktest"
      class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition mb-8 disabled:bg-gray-400 disabled:cursor-not-allowed"
    >
      {{ loading ? 'Running...' : `Run Backtest (${selectedStrategies.length} strategies)` }}
    </button>

    <div v-if="loading" class="mt-4 text-blue-600 flex items-center gap-2">
      <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
      Running backtest for {{ selectedStrategies.length }} strategies...
    </div>

    <div v-if="results" class="w-full max-w-6xl mt-10">
      <!-- Multi-Strategy Comparison Table -->
      <div v-if="results.multiStrategyResults && results.multiStrategyResults.comparisons.length > 1" class="mb-8 p-6 bg-white rounded-lg shadow-md">
        <h2 class="text-xl font-semibold mb-4">Multi-Strategy Performance Comparison</h2>
        <div class="overflow-x-auto">
          <table class="min-w-full table-auto">
            <thead>
              <tr class="bg-gray-50">
                <th class="px-4 py-2 text-left font-semibold">Strategy</th>
                <th class="px-4 py-2 text-right font-semibold">Total Return</th>
                <th class="px-4 py-2 text-right font-semibold">Outperformance</th>
                <th class="px-4 py-2 text-center font-semibold">vs Benchmark</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(comparison, index) in results.multiStrategyResults.comparisons" :key="index" class="border-t">
                <td class="px-4 py-2 font-medium">{{ comparison.strategy_name }}</td>
                <td class="px-4 py-2 text-right">
                  <span class="font-bold" :class="results.multiStrategyResults.strategies[index].summary.total_return >= 0 ? 'text-green-600' : 'text-red-600'">
                    {{ results.multiStrategyResults.strategies[index].summary.total_return }}%
                  </span>
                </td>
                <td class="px-4 py-2 text-right">
                  <span class="font-bold" :class="comparison.outperformance >= 0 ? 'text-green-600' : 'text-red-600'">
                    {{ comparison.outperformance > 0 ? '+' : '' }}{{ comparison.outperformance }}%
                  </span>
                </td>
                <td class="px-4 py-2 text-center">
                  <span v-if="comparison.beats_benchmark" class="text-green-600 font-semibold">✅ Beats</span>
                  <span v-else class="text-red-600 font-semibold">❌ Loses</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="mt-4 text-sm text-gray-600">
          Benchmark Return: <span class="font-semibold">{{ results.multiStrategyResults.benchmark.summary.total_return }}%</span>
          | Strategies Beating Benchmark: <span class="font-semibold">{{ results.multiStrategyResults.summary.strategies_beating_benchmark }}/{{ results.multiStrategyResults.summary.total_strategies }}</span>
        </div>
      </div>

      <!-- Single Strategy vs Benchmark Comparison -->
      <div v-else-if="results.comparison" class="mb-8 p-6 bg-white rounded-lg shadow-md">
        <h2 class="text-xl font-semibold mb-4">Strategy vs Benchmark Comparison</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="text-center p-4 bg-blue-50 rounded">
            <h3 class="font-semibold text-blue-800">Strategy Performance</h3>
            <p class="text-2xl font-bold text-blue-600">{{ results.strategy?.summary?.total_return || results.summary.total_return }}%</p>
          </div>
          <div class="text-center p-4 bg-gray-50 rounded">
            <h3 class="font-semibold text-gray-800">Benchmark (Buy & Hold)</h3>
            <p class="text-2xl font-bold text-gray-600">{{ results.benchmark.summary.total_return }}%</p>
          </div>
          <div class="text-center p-4 rounded" :class="results.comparison.strategy_beats_benchmark ? 'bg-green-50' : 'bg-red-50'">
            <h3 class="font-semibold" :class="results.comparison.strategy_beats_benchmark ? 'text-green-800' : 'text-red-800'">Outperformance</h3>
            <p class="text-2xl font-bold" :class="results.comparison.strategy_beats_benchmark ? 'text-green-600' : 'text-red-600'">
              {{ results.comparison.outperformance > 0 ? '+' : '' }}{{ results.comparison.outperformance }}%
            </p>
          </div>
        </div>
        <div class="mt-4 text-center">
          <p class="text-lg font-medium" :class="results.comparison.strategy_beats_benchmark ? 'text-green-600' : 'text-red-600'">
            {{ results.comparison.strategy_beats_benchmark ? '✅ Strategy beats benchmark!' : '❌ Strategy underperforms benchmark' }}
          </p>
        </div>
      </div>

      <!-- Combined Chart -->
      <div class="mb-8">
        <h2 class="text-xl font-semibold mb-4">
          {{ results.multiStrategyResults ? 'Multi-Strategy vs Benchmark Performance' : 'Strategy vs Benchmark Performance' }}
        </h2>
        <BacktestChart 
          :results="results" 
          :benchmark="results.benchmark" 
          :multiStrategyResults="results.multiStrategyResults"
        />
      </div>

      <!-- Strategy Results -->
      <div class="mb-8">
        <h2 class="text-xl font-semibold mb-4">Strategy Summary</h2>
        <ResultsSummary :summary="results.summary" />
      </div>

      <!-- Benchmark Results -->
      <div v-if="results.benchmark" class="mb-8">
        <h2 class="text-xl font-semibold mb-4">Benchmark Summary</h2>
        <ResultsSummary :summary="results.benchmark.summary" />
      </div>

      <!-- Trade Logs -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 class="text-lg font-semibold mb-2">Strategy Trade Log</h2>
          <pre class="bg-gray-100 p-2 rounded text-xs overflow-auto" style="max-height:250px;">
{{ JSON.stringify(results.trade_log, null, 2) }}
          </pre>
        </div>
        <div v-if="results.benchmark">
          <h2 class="text-lg font-semibold mb-2">Benchmark Trade Log</h2>
          <pre class="bg-gray-100 p-2 rounded text-xs overflow-auto" style="max-height:250px;">
{{ JSON.stringify(results.benchmark.trade_log, null, 2) }}
          </pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import BacktestChart from '../components/BacktestChart.vue'
import ResultsSummary from '../components/ResultsSummary.vue'

const loading = ref(false)
const csvString = ref('')
const pricesData = ref(null)
const results = ref(null)
const availableStrategies = ref([])
const selectedStrategies = ref([])

onMounted(async () => {
  try {
    // Fetch available strategies
    const strategiesResp = await fetch('http://localhost:8000/strategies')
    const strategiesResult = await strategiesResp.json()
    
    if (strategiesResult.strategies) {
      availableStrategies.value = strategiesResult.strategies
      // Select all strategies by default
      selectedStrategies.value = strategiesResult.strategies.map(s => s.endpoint)
      console.log("Loaded strategies:", availableStrategies.value)
    }

    // Fetch CSV data from API endpoint
    const resp = await fetch('http://localhost:8000/prices')
    const result = await resp.json()
    
    if (result.success) {
      // Set the CSV string for display
      csvString.value = result.csv_content
      
      // Use the parsed data directly from API
      pricesData.value = result.data.filter(x => x.date && x.ticker)
      
      console.log("Loaded price data from API:", pricesData.value)
    } else {
      console.error("Failed to load price data:", result.error)
      alert("Failed to load price data: " + result.error)
    }
  } catch (error) {
    console.error("Error fetching data:", error)
    alert("Error fetching data from API")
  }
})

function getStrategyName(endpoint) {
  const strategy = availableStrategies.value.find(s => s.endpoint === endpoint)
  return strategy ? strategy.name : endpoint
}

function removeStrategy(endpoint) {
  selectedStrategies.value = selectedStrategies.value.filter(s => s !== endpoint)
}

async function runBacktest() {
  loading.value = true
  results.value = null
  
  try {
    if (selectedStrategies.value.length > 1) {
      // Multiple strategies - use multi-strategy endpoint
      const resp = await fetch("http://localhost:8000/multi-strategy-backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          prices: pricesData.value,
          strategies: selectedStrategies.value,
          initial_cash: 10000
        })
      })
      const multiResults = await resp.json()
      
      // Show the first strategy's results in the existing UI but include multi-strategy data
      if (multiResults.strategies && multiResults.strategies.length > 0) {
        const firstStrategy = multiResults.strategies[0]
        results.value = {
          ...firstStrategy,
          benchmark: multiResults.benchmark,
          comparison: {
            outperformance: multiResults.comparisons[0]?.outperformance || 0,
            strategy_beats_benchmark: multiResults.comparisons[0]?.beats_benchmark || false
          },
          // Add multi-strategy data for comparison table
          multiStrategyResults: multiResults
        }
      }
    } else {
      // Single strategy - use regular backtest endpoint for backward compatibility
      const resp = await fetch("http://localhost:8000/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prices: pricesData.value })
      })
      results.value = await resp.json()
    }
  } catch (err) {
    console.error("Backtest error:", err)
    alert("Backtest failed")
  } finally {
    loading.value = false
  }
}
</script>
