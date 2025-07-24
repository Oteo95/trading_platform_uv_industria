<template>
  <div class="w-full max-w-lg mb-6">
    <label class="block mb-2 text-sm font-medium text-gray-700">Select CSV Historical Data</label>
    <input
      type="file"
      accept=".csv"
      @change="handleFile"
      class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer focus:outline-none"
    />
  </div>
</template>

<script setup>
import Papa from 'papaparse'
const emit = defineEmits(['data-selected'])

function handleFile(event) {
  const file = event.target.files[0]
  if (!file) return
  Papa.parse(file, {
    header: true,
    complete: (results) => {
      emit('data-selected', results.data)
    },
    error: () => alert('Error reading CSV file.')
  })
}
</script>
