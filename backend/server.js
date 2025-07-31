const express = require("express");
const connectDB = require("./utils/db");
const uploadRoute = require("./routes/upload");
require("dotenv").config();

const app = express();
app.use(express.json());
app.use('/api', uploadRoute);

const PORT = process.env.PORT || 3000;

connectDB().then(() => {
  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
  });
});
