<template>
  <div class="bg-white rounded-xl shadow p-4 flex flex-col gap-2">
    <h2 class="text-xl font-semibold mb-2">Summary</h2>
    <div v-if="summary">
      <div><b>Total Return:</b> {{ summary.total_return }}%</div>
      <div><b>Annualized Return:</b> {{ summary.annualized_return }}%</div>
      <div><b>Max Drawdown:</b> {{ summary.max_drawdown }}%</div>
      <div><b>Sharpe Ratio:</b> {{ summary.sharpe_ratio }}</div>
    </div>
    <div v-else class="text-gray-400">No summary data.</div>
  </div>
</template>

<script setup>
const props = defineProps({
  summary: Object
})
</script>
