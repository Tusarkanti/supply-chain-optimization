import React, { useState } from "react";
import { jsPDF } from "jspdf";
import * as XLSX from "xlsx";
import ApiService from '../api';
import "./Reporting.css";

function Reporting({ token }) {
  const [reportType, setReportType] = useState("inventory");
  const [dateRange, setDateRange] = useState("30");
  const [loading, setLoading] = useState(false);
  const [reportData, setReportData] = useState(null);

  // Fetch report from backend
  const handleGenerateReport = async () => {
    setLoading(true);
    setReportData(null);
    try {
      const data = await ApiService.generateReport(reportType, {
        date_range: dateRange,
        format: "json",
      }, token);
      setReportData(data);
    } catch (err) {
      console.error("Error generating report:", err);
      alert("Failed to generate report. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Export Handlers
  const handleExport = (format) => {
    if (!reportData) {
      alert("Please generate a report first!");
      return;
    }

    switch (format) {
      case "pdf":
        exportPDF();
        break;
      case "excel":
        exportExcel();
        break;
      case "csv":
        exportCSV();
        break;
      default:
        break;
    }
  };

  const exportPDF = () => {
    const doc = new jsPDF();
    doc.text(`${reportType.toUpperCase()} REPORT`, 10, 10);
    doc.setFontSize(10);
    doc.text(JSON.stringify(reportData, null, 2), 10, 20);
    doc.save(`${reportType}_report.pdf`);
  };

  const exportExcel = () => {
    const worksheet = XLSX.utils.json_to_sheet(reportData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Report");
    XLSX.writeFile(workbook, `${reportType}_report.xlsx`);
  };

  const exportCSV = () => {
    const worksheet = XLSX.utils.json_to_sheet(reportData);
    const csvOutput = XLSX.utils.sheet_to_csv(worksheet);
    const blob = new Blob([csvOutput], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${reportType}_report.csv`;
    link.click();
  };

  return (
    <div className="reporting">
      <div className="reporting-header">
        <h2>📊 Advanced Reporting Dashboard</h2>
        <p>Generate, preview, and export detailed supply chain reports</p>
      </div>

      {/* Report Controls */}
      <div className="report-controls">
        <div className="control-group">
          <label>Report Type:</label>
          <select
            value={reportType}
            onChange={(e) => setReportType(e.target.value)}
          >
            <option value="inventory">Inventory Report</option>
            <option value="demand">Demand Forecast</option>
            <option value="logistics">Logistics Performance</option>
            <option value="financial">Financial Summary</option>
          </select>
        </div>

        <div className="control-group">
          <label>Date Range:</label>
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
          >
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
            <option value="365">Last year</option>
          </select>
        </div>

        <button
          onClick={handleGenerateReport}
          disabled={loading}
          className="generate-button"
        >
          {loading ? "⏳ Generating..." : "🚀 Generate Report"}
        </button>
      </div>

      {/* Export Controls */}
      <div className="export-controls">
        <button onClick={() => handleExport("pdf")} className="export-button">
          📄 Export PDF
        </button>
        <button onClick={() => handleExport("excel")} className="export-button">
          📊 Export Excel
        </button>
        <button onClick={() => handleExport("csv")} className="export-button">
          📑 Export CSV
        </button>
      </div>

      {/* Report Preview */}
      {reportData && (
        <div className="report-preview">
          <h3>Report Preview</h3>
          <table>
            <thead>
              <tr>
                {Object.keys(reportData[0] || {}).map((key) => (
                  <th key={key}>{key.toUpperCase()}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {reportData.map((row, idx) => (
                <tr key={idx}>
                  {Object.values(row).map((val, i) => (
                    <td key={i}>{val?.toString()}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default Reporting;
