const axios = require('axios');

async function triggerPythonService(fileId) {
  try {
    await axios.post(process.env.PYTHON_SERVICE_URL, { fileId });
    console.log("Python microservice triggered");
  } catch (error) {
    console.error("Failed to trigger Python service:", error.message);
  }
}

module.exports = triggerPythonService;
