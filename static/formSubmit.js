$("#submit").click(function () {
    var q1 = $('#ques1').val();
    var q2 = $('#ques2').val();
    $.ajax({
        url: "predict",
        data: {
            "ques1": q1,
            "ques2": q2
        },
        type: "GET",
        success: function (feature_list) {
            var predictOutput = new Array();
            predictOutput.push(["Feature Set", "Logisitic Regression", "XG_Boost"]);
            predictOutput.push([1, feature_list[0]['Logistic Regression'][0], feature_list[0]['XG_Boost'][0]]);
            predictOutput.push([2, feature_list[0]['Logistic Regression'][1], feature_list[0]['XG_Boost'][1]]);
            predictOutput.push([3, feature_list[0]['Logistic Regression'][2], feature_list[0]['XG_Boost'][2]]);
            predictOutput.push([4, feature_list[0]['Logistic Regression'][3], feature_list[0]['XG_Boost'][3]]);
            predictOutput.push([5, feature_list[0]['Logistic Regression'][4], feature_list[0]['XG_Boost'][4]]);
            predictOutput.push([6, feature_list[0]['Logistic Regression'][5], feature_list[0]['XG_Boost'][5]]);
            //Create a HTML Table element.
            var table = $("<table />");
            table.addClass("u-full-width");

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
                    cell.attr("align", "center");
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
                            {y: parseFloat(feature_list[1]["fuzz_qratio"]), label: "fuzz_qratio"},
                            {y: parseFloat(feature_list[1]["fuzz_WRatio"]), label: "fuzz_WRatio"},
                            {y: parseFloat(feature_list[1]["wmd"]), label: "wmd"},
                            {y: parseFloat(feature_list[1]["norm_wmd"]), label: "norm_wmd"},
                            {y: parseFloat(feature_list[1]["cosine_distance"]), label: "cosine_distance"},
                            {y: parseFloat(feature_list[1]["euclidean_distance"]), label: "euclidean_distance"},
                            {y: parseFloat(feature_list[1]["braycurtis_distance"]), label: "braycurtis_distance"},
                            {y: parseFloat(feature_list[1]["cosSim"]), label: "cosSim"},
                            {y: parseFloat(feature_list[1]["jaccard_distance"]), label: "jaccard_distance"}
                        ]
                    }
                ]
            });
            chart.render();
        }
    });
});
