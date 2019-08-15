function ShowLoading()
{
    $("#overlay").show();
    $("#loading").show();
}
function HideLoading()
{
    $("#overlay").hide();
    $("#loading").hide();
}
function CheckAccessToken()
{
    var hash = window.location.hash;
    console.log(hash)
    $("#token").val(hash)
    $( "form" ).submit();
}
