const express = require("express");
const multer = require("multer");
const router = express.Router();
const {
  uploadToGridFS,
  deleteFileById,
  deleteAllFiles,
} = require("../services/mongoUploader");

const storage = multer.memoryStorage();
const upload = multer({ storage });

router.post("/upload", upload.single("file"), uploadToGridFS, (req, res) => {
  res.status(200).json({
    message: "✅ File uploaded successfully",
    fileId: req.fileId,
    filename: req.storedName,
  });
});

router.delete("/delete/:id", deleteFileById);
router.delete("/delete", deleteAllFiles);

module.exports = router;
