
var scotchTodo = angular.module('scotchTodo', []);

function mainController($scope, $http) {
    $scope.formData = {};
    $scope.IsVisible = false;
    // when landing on the page, get all todos and show them
    $http.get('/api/todos')
        .success(function(data) {
            $scope.todos = data;
            console.log(data);
        })
        .error(function(data) {
            console.log('Error: ' + data);
        });

    // when submitting the add form, send the text to the node API
    $scope.createTodo = function() {
        $http.post('/api/todos', $scope.formData)
            .success(function(data) {
                $scope.formData = {}; // clear the form so our user is ready to enter another
                $scope.todos = data;
                console.log(data);
            })
            .error(function(data) {
                console.log('Error: ' + data);
            });
    };

    // delete a todo after checking it
    $scope.deleteTodo = function(id) {
        $http.delete('/api/todos/' + id)
            .success(function(data) {
                $scope.todos = data;
                console.log(data);
            })
            .error(function(data) {
                console.log('Error: ' + data);
            });
    };
    $scope.detectTodo = function() {
        $scope.IsVisible= true;
        var display;
        $http.post('/api/ml', $scope.formData)
            .success(function(data) {
                console.log('Detector is working');
                // var t = data.toString();
                $scope.formData = {}; // clear the form so our user is ready to enter another
                // $scope.todos = data;
                console.log(data);
                // $scope.formData.text = data;
                display = data;
                data = data.replace("\"","");
                data = data.replace("\"","");
                var output1 = document.getElementById('output1');
                output1.innerHTML = data;
                $scope.IsVisible= false;
                // alert(data)
            })
            .error(function(data) {
                console.log('Error is: ' + data);
            });
            // console.log(data);
    };
    $scope.classifyTodo = function () {
        // var display ='';
        $scope.IsVisible= true;
        $http.post('/api/clf', $scope.formData)
            .success(function(data) {
                console.log('Classifier is also working');
                // var t = data.toString();
                $scope.formData = {}; // clear the form so our user is ready to enter another
                // $scope.todos = data;
                console.log(data);
                // $scope.formData.text = data;
                // display += '\n' + data;
                $scope.IsVisible= false;
                // alert(data);
                data = data.replace("\"","");
                data = data.replace("\"","");
                var output2 = document.getElementById('output2');
                output2.innerHTML = data;

            })
            .error(function(data) {
                console.log('Error is: ' + data);
            });
        // console.log(data);
    }

}