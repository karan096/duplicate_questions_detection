/**
 * Created by naivedya on 26/11/17.
 */
$("#submit").click(function () {
    $.ajax({
        url: "predict", success: function (result) {
            //alert(result);
            var predictOutput = new Array();
            predictOutput.push(["Feature Set", "Logisitic Regression", "XG_Boost"]);
            predictOutput.push([1, "John Hammond", "United States"]);
            predictOutput.push([2, "Mudassar Khan", "India"]);
            predictOutput.push([3, "Suzanne Mathews", "France"]);
            predictOutput.push([4, "Robert Schidner", "Russia"]);
            predictOutput.push([5, "HI_hop_in", "India"]);
            predictOutput.push([6, "JK_LOL", "China"]);
            //Create a HTML Table element.
            var table = $("<table />");
            table[0].border = "1";

            //Get the count of columns.
            var columnCount = predictOutput[0].length;

            //Add the header row.
            var row = $(table[0].insertRow(-1));
            for (var i = 0; i < columnCount; i++) {
                var headerCell = $("<th />");
                headerCell.html(predictOutput[0][i]);
                row.append(headerCell);
            }

            //Add the data rows.
            for (var i = 1; i < predictOutput.length; i++) {
                row = $(table[0].insertRow(-1));
                for (var j = 0; j < columnCount; j++) {
                    var cell = $("<td />");
                    cell.html(predictOutput[i][j]);
                    row.append(cell);
                }
            }

            var dvTable = $("#dvTable");
            $("#result").html("Result: ")
            dvTable.html("");
            dvTable.append(table);
            var chart = new CanvasJS.Chart("chartContainer",
            {
                title: {
                    text: "Comparison of Question-Pair features: "
                },
                data: [
                    {
                        type: "bar",
                        dataPoints: [
                            {y: 198, label: "Italy"},
                            {y: 201, label: "China"},
                            {y: 202, label: "France"},
                            {y: 236, label: "Great Britain"},
                            {y: 395, label: "Soviet Union"},
                            {y: 857, label: "USA"}
                        ]
                    }
                ]
            });
            chart.render();
        }
    });
});